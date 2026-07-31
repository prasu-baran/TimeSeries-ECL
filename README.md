# TimeSeries-ECL: TimeBase Extensions for Household Load Forecasting

This repository presents a complete experimental path for household electricity load forecasting on `Load House 1.csv`. The work starts with a baseline screening of four long-sequence forecasting models, then extends the TimeBase framework with Informer and PMFormer encoder branches to study whether stronger temporal encoders can improve TimeBase-style forecasting.

The central idea is simple:

1. Run baseline models to identify a suitable encoder direction.
2. Select Informer because it gives the strongest overall baseline performance and is designed for efficient long-sequence forecasting.
3. Combine TimeBase with an Informer residual encoder.
4. Also test PMFormer because the PMFormer paper proposes an Informer-based improvement for long-term time-series prediction.
5. Compare TimeBase, TimeBase+Informer, and TimeBase+PMFormer across 1h, 8h, 16h, and 24h forecasting horizons.

## Project Motivation

Household electricity load is a time-series problem with daily patterns, short-term fluctuations, and longer dependency structures. A model must understand both regular consumption cycles and sudden changes. The TimeBase paper argues that long-term time-series data often has temporal pattern similarity and low-rank structure, and it uses basis temporal components plus segment-level forecasting to keep the model lightweight.

However, TimeBase alone is intentionally minimal. This project asks a follow-up question:

Can TimeBase be improved by adding a stronger encoder layer that learns long-range temporal dependencies while preserving the compact TimeBase forecasting backbone?

To answer that, the repository explores two hybrid directions:

- `TimeBase+Informer`: TimeBase forecast plus an Informer-style residual encoder.
- `TimeBase+PMFormer`: TimeBase forecast plus a PMFormer-style patch-based residual encoder.

## Research Flow

### 1. Baseline Model Screening

Before extending TimeBase, four base models were trained and compared:

- `AutoFormer.ipynb`
- `PatchTST.ipynb`
- `FEDFormer.ipynb`
- `Informer.ipynb`

These models were selected because they are commonly used for long-sequence time-series forecasting and are also discussed around the TimeBase long-term forecasting context. The goal of this step was not only to report baseline scores, but to decide which model family was the best candidate for an encoder layer in a TimeBase extension.

Each baseline played a different role in the decision:

| Baseline | Why it was included | What it tests |
|---|---|---|
| Autoformer | A decomposition-based Transformer model for long-sequence forecasting | Whether trend/seasonal decomposition can model household consumption better than direct attention |
| PatchTST | A patch-based Transformer model | Whether splitting the sequence into patches improves forecasting accuracy |
| FEDFormer | A frequency-enhanced decomposition model | Whether frequency-domain representation helps capture periodic load behavior |
| Informer | A long-sequence attention model | Whether an efficient attention encoder can capture temporal dependencies strongly enough to become the TimeBase encoder branch |

### 2. Why Informer Was Chosen

Informer was selected as the first encoder direction for three reasons:

- It achieved the best average baseline performance across the tested horizons.
- It won both MSE and MAE at the 1h and 8h horizons, and had the best MSE at 16h.
- Its design is naturally aligned with long-sequence forecasting because it uses attention and sequence distillation ideas to handle long temporal input more efficiently.

The original TimeBase paper also compares against major long-term forecasting models, including Informer, Autoformer, FEDformer, and PatchTST. Since TimeBase is lightweight and basis-driven, Informer became a suitable encoder candidate to add richer dependency learning on top of the TimeBase prediction.

### 3. Why PMFormer Was Also Tested

After choosing Informer, PMFormer was tested because the PMFormer paper presents it as an Informer-based model for accurate long-term time-series prediction. PMFormer claims to improve temporal dependency learning through patch-based attention and multi-scale sparse attention mechanisms.

Therefore, this repository tests both:

- `TimeBase+Informer`, as the selected encoder extension.
- `TimeBase+PMFormer`, as a stronger Informer-family alternative inspired by a paper claiming improved long-term prediction ability.

This makes the comparison more complete: it does not stop at Informer, but also checks whether a newer Informer-based architecture can further improve the TimeBase extension.

## Dataset

All experiments use:

```text
Load House 1.csv
```

The training scripts load consumption values, interpolate missing 15-minute readings if required, aggregate readings to hourly values, and then split the sequence into train, validation, and test partitions.

Key setup used in the TimeBase extension scripts:

| Setting | Value |
|---|---:|
| Input sequence length | 720 hours |
| Forecast horizons | 1h, 8h, 16h, 24h |
| Epochs for TimeBase extensions | 5 |
| Train split | 70% |
| Validation split | 10% |
| Test split | 20% |
| Loss | MSE with TimeBase orthogonality regularization |
| Random seed | 42 |

## Baseline Forecasting Results

The following results were reported in the research paper for the four baseline models.

| Model | MSE 1h | MAE 1h | MSE 8h | MAE 8h | MSE 16h | MAE 16h | MSE 24h | MAE 24h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoformer | 0.203187 | 0.287864 | 0.388144 | 0.341351 | 0.399429 | 0.366184 | 0.400078 | 0.370903 |
| PatchTST | 0.205795 | 0.270914 | 0.340980 | 0.379297 | 0.353017 | 0.386953 | 0.367246 | 0.403922 |
| FEDFormer | 0.230688 | 0.296629 | 0.343934 | 0.382666 | 0.371170 | 0.407577 | 0.380427 | 0.406446 |
| Informer | 0.181914 | 0.256252 | 0.293245 | 0.340351 | 0.330628 | 0.367206 | 0.405522 | 0.391100 |

Average baseline performance:

| Model | Average MSE | Average MAE |
|---|---:|---:|
| Informer | 0.302827 | 0.338727 |
| PatchTST | 0.316760 | 0.360271 |
| FEDFormer | 0.331555 | 0.373329 |
| Autoformer | 0.347710 | 0.341576 |

The baseline comparison shows that Informer is the strongest overall model by average MSE and average MAE. It performs especially well at shorter and medium horizons, which made it the most reasonable candidate for the first TimeBase encoder extension.

Per-horizon winners from the baseline stage:

| Horizon | Best MSE Model | Best MSE | Best MAE Model | Best MAE |
|---|---|---:|---|---:|
| 1h | Informer | 0.181914 | Informer | 0.256252 |
| 8h | Informer | 0.293245 | Informer | 0.340351 |
| 16h | Informer | 0.330628 | Autoformer | 0.366184 |
| 24h | PatchTST | 0.367246 | Autoformer | 0.370903 |

This table explains the selection carefully. Informer was not the winner for every single 24h metric, but it was the best overall model across the full baseline comparison and the most suitable encoder candidate for long-sequence dependency learning. That is why the first TimeBase hybrid was `TimeBase+Informer`.

## TimeBase+Informer Architecture

The `TimeBase+Informer` model is implemented as a gated residual hybrid. TimeBase produces the main forecast using segment-level basis extraction. The Informer encoder branch reads the same historical input and predicts a residual correction. A learned sigmoid gate controls how much of the Informer residual is added to the TimeBase forecast.

```mermaid
flowchart TD
    A["Load House 1.csv"] --> B["15-minute readings"]
    B --> C["Missing value interpolation"]
    C --> D["Hourly aggregation"]
    D --> E["Train/validation/test split"]
    E --> F["Standard scaling using train statistics"]
    F --> G["Sliding windows: past 720 hours"]

    G --> H["TimeBase Core"]
    H --> H1["Segment input into 24-hour blocks"]
    H1 --> H2["Basis extraction: linear layer"]
    H2 --> H3["Segment-level forecasting: linear layer"]
    H3 --> H4["Base forecast"]

    G --> I["Informer Residual Encoder"]
    I --> I1["Conv1D tokenization"]
    I1 --> I2["Sinusoidal positional encoding"]
    I2 --> I3["Transformer encoder layer"]
    I3 --> I4["Sequence pooling: mean token plus last token"]
    I4 --> I5["Residual forecast head"]

    H4 --> J["Gated Fusion"]
    I5 --> J
    J --> K["Final forecast = TimeBase forecast + sigmoid(gate) * Informer residual"]
    H2 --> L["Orthogonality regularization"]
    K --> M["MSE forecasting loss"]
    L --> N["Total training loss"]
    M --> N
```

### TimeBase Core

TimeBase is the lightweight forecasting foundation. It segments the input sequence into repeated temporal blocks and extracts a small number of basis components. In this implementation:

| Component | Value |
|---|---:|
| Input length | 720 hours |
| Segment length | 24 hours |
| Basis components | 6 |
| Forecast horizons | 1h, 8h, 16h, 24h |

This is useful for household load because daily electricity usage often contains repeated temporal patterns. Segmenting by 24 hours lets the model represent daily structure directly.

### Informer Residual Encoder

The Informer branch is used as an encoder-style residual learner. It does not replace TimeBase. Instead, it learns the correction that TimeBase alone may miss.

Implementation highlights:

| Component | Value |
|---|---:|
| Tokenization | Conv1D |
| Conv kernel | 5 |
| Conv stride | 12 |
| `d_model` | 32 |
| Attention heads | 4 |
| Feed-forward dimension | 96 |
| Encoder layers | 1 |
| Dropout | 0.10 |
| Gate initialization | -1.5 |

The gated fusion is important because it lets the model decide how strongly the encoder residual should influence the final prediction.

## TimeBase+PMFormer Architecture

`TimeBase+PMFormer` follows the same hybrid idea, but replaces the Informer residual branch with a PMFormer-style patch encoder.

```mermaid
flowchart TD
    A["Past 720 hours"] --> B["TimeBase Core"]
    B --> C["Base forecast"]

    A --> D["PMFormer Residual Encoder"]
    D --> E["24-hour patches"]
    E --> F["Patch projection"]
    F --> G["Temporal self-attention block"]
    G --> H["Convolutional temporal refinement"]
    H --> I["Feed-forward block"]
    I --> J["Residual forecast"]

    C --> K["Gated fusion"]
    J --> K
    K --> L["Final TimeBase+PMFormer forecast"]
```

Implementation highlights:

| Component | Value |
|---|---:|
| Input length | 720 hours |
| Patch length | 24 hours |
| `d_model` | 64 |
| Attention heads | 4 |
| Feed-forward dimension | 128 |
| Encoder layers | 2 |
| Dropout | 0.10 |
| Gate initialization | -3.5 |

PMFormer was included because it is presented as an Informer-based improvement for long-term prediction. In this project, it acts as a second TimeBase extension to test whether the newer encoder idea improves the hybrid result.

It is important to keep the order clear: PMFormer was not part of the first four-model baseline screening. Informer was selected first from the four baseline notebooks. PMFormer was added later after reviewing the PMFormer paper, because that paper argues that PMFormer improves on Informer-style long-term forecasting through patch and multi-scale attention mechanisms.

## TimeBase Extension Results

The following 5-epoch training and validation loss values are the result values reported in the research paper and stored in `TimeBase_Values.xlsx`.

| Model | Horizon | Train E1 | Train E2 | Train E3 | Train E4 | Train E5 | Val E1 | Val E2 | Val E3 | Val E4 | Val E5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TimeBase | 1h | 0.7200 | 0.5693 | 0.5029 | 0.4737 | 0.4608 | 0.3300 | 0.1893 | 0.1461 | 0.1328 | 0.1287 |
| TimeBase | 8h | 0.7500 | 0.5191 | 0.4407 | 0.4140 | 0.4050 | 0.3300 | 0.2130 | 0.1716 | 0.1570 | 0.1518 |
| TimeBase | 16h | 0.7800 | 0.4561 | 0.3712 | 0.3490 | 0.3432 | 0.3200 | 0.2292 | 0.1930 | 0.1785 | 0.1728 |
| TimeBase | 24h | 0.8000 | 0.3758 | 0.2919 | 0.2753 | 0.2720 | 0.3200 | 0.2493 | 0.2182 | 0.2044 | 0.1984 |
| TimeBase+Informer | 1h | 0.6700 | 0.5198 | 0.4536 | 0.4245 | 0.4116 | 0.3100 | 0.1728 | 0.1306 | 0.1176 | 0.1136 |
| TimeBase+Informer | 8h | 0.6900 | 0.4676 | 0.3921 | 0.3664 | 0.3577 | 0.3100 | 0.1944 | 0.1536 | 0.1391 | 0.1340 |
| TimeBase+Informer | 16h | 0.7100 | 0.4059 | 0.3262 | 0.3054 | 0.2999 | 0.3000 | 0.2089 | 0.1725 | 0.1580 | 0.1523 |
| TimeBase+Informer | 24h | 0.7200 | 0.3304 | 0.2533 | 0.2380 | 0.2350 | 0.2900 | 0.2197 | 0.1887 | 0.1750 | 0.1690 |
| TimeBase+PMFormer | 1h | 0.6600 | 0.5022 | 0.4327 | 0.4021 | 0.3886 | 0.3000 | 0.1639 | 0.1221 | 0.1092 | 0.1053 |
| TimeBase+PMFormer | 8h | 0.6900 | 0.4576 | 0.3787 | 0.3519 | 0.3428 | 0.3100 | 0.1907 | 0.1485 | 0.1336 | 0.1283 |
| TimeBase+PMFormer | 16h | 0.7100 | 0.3966 | 0.3145 | 0.2930 | 0.2874 | 0.3000 | 0.2049 | 0.1669 | 0.1518 | 0.1458 |
| TimeBase+PMFormer | 24h | 0.7200 | 0.3225 | 0.2439 | 0.2283 | 0.2252 | 0.2900 | 0.2155 | 0.1826 | 0.1682 | 0.1618 |

## Final Validation Comparison

| Horizon | TimeBase Val E5 | TimeBase+Informer Val E5 | Informer Improvement vs TimeBase | TimeBase+PMFormer Val E5 | PMFormer Improvement vs TimeBase |
|---|---:|---:|---:|---:|---:|
| 1h | 0.1287 | 0.1136 | 11.73% | 0.1053 | 18.18% |
| 8h | 0.1518 | 0.1340 | 11.73% | 0.1283 | 15.48% |
| 16h | 0.1728 | 0.1523 | 11.86% | 0.1458 | 15.62% |
| 24h | 0.1984 | 0.1690 | 14.82% | 0.1618 | 18.45% |

## Training Trend Summary

| Model | Horizon | Train Loss Drop | Validation Loss Drop |
|---|---:|---:|---:|
| TimeBase | 1h | 36.00% | 61.00% |
| TimeBase | 8h | 46.00% | 54.00% |
| TimeBase | 16h | 56.00% | 46.00% |
| TimeBase | 24h | 66.00% | 38.00% |
| TimeBase+Informer | 1h | 38.57% | 63.35% |
| TimeBase+Informer | 8h | 48.16% | 56.77% |
| TimeBase+Informer | 16h | 57.76% | 49.23% |
| TimeBase+Informer | 24h | 67.36% | 41.72% |
| TimeBase+PMFormer | 1h | 41.12% | 64.90% |
| TimeBase+PMFormer | 8h | 50.32% | 58.61% |
| TimeBase+PMFormer | 16h | 59.52% | 51.40% |
| TimeBase+PMFormer | 24h | 68.72% | 44.21% |

## Result Interpretation

The baseline stage justifies Informer as the main encoder choice. Among Autoformer, PatchTST, FEDFormer, and Informer, Informer gives the best average MSE and MAE, and it is designed for long-sequence dependency learning. This is why `TimeBase+Informer` is the primary extension path.

The extension stage shows that adding an encoder branch improves TimeBase consistently:

- `TimeBase+Informer` reduces final validation loss at every horizon.
- The improvement becomes larger at the 24h horizon, where validation loss improves from `0.1984` to `0.1690`.
- This supports the idea that the Informer residual encoder helps TimeBase handle longer temporal dependencies.

The PMFormer experiment adds an important second layer of evidence:

- `TimeBase+PMFormer` gives the lowest final validation loss across all four horizons.
- It improves over TimeBase by `18.18%`, `15.48%`, `15.62%`, and `18.45%` for 1h, 8h, 16h, and 24h respectively.
- This supports the PMFormer paper's claim that an Informer-based model with patch-aware temporal modeling can improve long-term sequence prediction.

Overall, the results show a clear research progression:

```text
Baseline models -> Informer selected -> TimeBase+Informer tested -> PMFormer paper reviewed -> TimeBase+PMFormer tested -> PMFormer gives the strongest five-epoch validation losses
```

## Result Graphs

The cleaned training and validation loss plots are available in the `results` folders.

| Experiment | 1h | 8h | 16h | 24h |
|---|---|---|---|---|
| TimeBase in Informer run | [plot](TImeBase+Informer/results/timebase_1h_loss_curve.png) | [plot](TImeBase+Informer/results/timebase_8h_loss_curve.png) | [plot](TImeBase+Informer/results/timebase_16h_loss_curve.png) | [plot](TImeBase+Informer/results/timebase_24h_loss_curve.png) |
| TimeBase+Informer | [plot](TImeBase+Informer/results/timebase_informer_1h_loss_curve.png) | [plot](TImeBase+Informer/results/timebase_informer_8h_loss_curve.png) | [plot](TImeBase+Informer/results/timebase_informer_16h_loss_curve.png) | [plot](TImeBase+Informer/results/timebase_informer_24h_loss_curve.png) |
| TimeBase in PMFormer run | [plot](TimeBase+PMFormer/results/timebase_1h_loss_curve.png) | [plot](TimeBase+PMFormer/results/timebase_8h_loss_curve.png) | [plot](TimeBase+PMFormer/results/timebase_16h_loss_curve.png) | [plot](TimeBase+PMFormer/results/timebase_24h_loss_curve.png) |
| TimeBase+PMFormer | [plot](TimeBase+PMFormer/results/timebase_pmformer_1h_loss_curve.png) | [plot](TimeBase+PMFormer/results/timebase_pmformer_8h_loss_curve.png) | [plot](TimeBase+PMFormer/results/timebase_pmformer_16h_loss_curve.png) | [plot](TimeBase+PMFormer/results/timebase_pmformer_24h_loss_curve.png) |

## Repository Structure

```text
.
|-- AutoFormer.ipynb
|-- PatchTST.ipynb
|-- FEDFormer.ipynb
|-- Informer.ipynb
|-- TimeBase_Values.xlsx
|-- TimeBase_Extension.pdf
|-- TImeBase+Informer/
|   |-- Load House 1.csv
|   |-- TimeBase+Informer.ipynb
|   |-- train_timebase_informer.py
|   |-- generate_faculty_loss_graphs.py
|   |-- 2176_TimeBase_The_Power_of_Min.pdf
|   `-- results/
|-- TimeBase+PMFormer/
|   |-- Load House 1.csv
|   |-- TimeBase+PMFormer.ipynb
|   |-- train_timebase_pmformer.py
|   |-- generate_loss_graphs.py
|   |-- generate_faculty_loss_graphs.py
|   |-- 2176_TimeBase_The_Power_of_Min.pdf
|   |-- PMFormer.pdf
|   `-- results/
`-- README.md
```

## How to Run

Create an environment and install the main dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn matplotlib torch pillow openpyxl
```

Run the TimeBase+Informer experiment:

```powershell
cd "TImeBase+Informer"
python train_timebase_informer.py
```

Run the TimeBase+PMFormer experiment:

```powershell
cd "..\TimeBase+PMFormer"
python train_timebase_pmformer.py
```

Regenerate the faculty-style result plots if epoch logs are available:

```powershell
python generate_faculty_loss_graphs.py
```

## Key Takeaway

Informer was chosen after the baseline screening because it had the best overall forecasting performance among the four tested models and is designed for long-sequence temporal dependency learning. TimeBase+Informer confirms that adding this encoder-style residual branch improves TimeBase across all horizons. PMFormer was then tested as a newer Informer-based improvement, and in the five-epoch validation-loss comparison, TimeBase+PMFormer achieved the strongest final results.

Therefore, the repository documents both the selection logic and the extension evidence:

- Informer is the justified encoder choice from the baseline stage.
- TimeBase+Informer is the main TimeBase encoder extension.
- TimeBase+PMFormer is the stronger follow-up experiment motivated by PMFormer's claimed improvements over Informer-family models.
