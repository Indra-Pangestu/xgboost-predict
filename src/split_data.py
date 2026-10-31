"""Create duplicate-safe 80/20 train-test partitions from cleaned data."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data" / "processed" / "data_cleaned.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORT = PROJECT_ROOT / "outputs" / "reports" / "split_report.md"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
FEATURES = ["Soil Moisture", "Temperature", "Usia Tanam"]
TARGET = "Volume Penyiraman"
GROUP_COLUMNS = FEATURES + [TARGET]

# Allow direct execution from either repository root or src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualizations import generate_scatter_plots  # noqa: E402


def run_split() -> tuple[int, int]:
    data = pd.read_csv(SOURCE)
    required = GROUP_COLUMNS + ["source_excel_row"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[GROUP_COLUMNS].isna().any().any():
        raise ValueError("Cannot group records with missing dataset values")

    # Each identical (X, y) record tuple receives one group ID.
    groups = data.groupby(GROUP_COLUMNS, sort=False, dropna=False).ngroup()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(data[FEATURES], data[TARGET], groups=groups))
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    train_groups = set(groups.iloc[train_idx].tolist())
    test_groups = set(groups.iloc[test_idx].tolist())
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"Duplicate groups cross partitions: {sorted(overlap)}")
    if TARGET in FEATURES:
        raise AssertionError("Target must not be included among model features")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES / "splits").mkdir(parents=True, exist_ok=True)
    train[FEATURES].to_csv(SPLIT_DIR / "X_train.csv", index=False, encoding="utf-8-sig")
    test[FEATURES].to_csv(SPLIT_DIR / "X_test.csv", index=False, encoding="utf-8-sig")
    train[[TARGET]].to_csv(SPLIT_DIR / "y_train.csv", index=False, encoding="utf-8-sig")
    test[[TARGET]].to_csv(SPLIT_DIR / "y_test.csv", index=False, encoding="utf-8-sig")

    # Counts chart.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.facecolor": "#FBFCFE", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(["Training", "Testing"], [len(train), len(test)],
                  color=["#2673A6", "#E39C37"], width=0.58)
    ax.set_title("Jumlah Data Training dan Testing", weight="semibold", pad=12)
    ax.set_ylabel("Jumlah baris")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.bar_label(bars, padding=4)
    fig.tight_layout()
    fig.savefig(FIGURES / "splits" / "training_testing_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Target distributions use the actual split target values.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(train[TARGET], bins=24, alpha=0.62, color="#2673A6", label=f"Training (n={len(train)})")
    ax.hist(test[TARGET], bins=24, alpha=0.62, color="#E39C37", label=f"Testing (n={len(test)})")
    ax.set_title("Distribusi Volume Penyiraman", weight="semibold", pad=12)
    ax.set_xlabel("Volume Penyiraman (mL)")
    ax.set_ylabel("Frekuensi")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "splits" / "target_distribution_train_test.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Three scatter plots per partition, points only; helper performs plot-only numeric conversion.
    train_plots = generate_scatter_plots(train, "train")
    test_plots = generate_scatter_plots(test, "test")

    # Verify no exact (feature, target) tuple appears in both outputs.
    train_tuples = set(map(tuple, train[GROUP_COLUMNS].itertuples(index=False, name=None)))
    test_tuples = set(map(tuple, test[GROUP_COLUMNS].itertuples(index=False, name=None)))
    assert not (train_tuples & test_tuples)
    assert TARGET not in FEATURES
    assert len(train) + len(test) == len(data)

    report_lines = [
        "# Laporan Train-Test Split", "",
        "**Sumber:** `data/processed/data_cleaned.csv`.",
        "**Metode:** `GroupShuffleSplit(test_size=0.20, random_state=42)`; grup dibentuk dari seluruh empat kolom dataset.", "",
        "## Hasil", "",
        f"- Baris sumber: **{len(data)}**.",
        f"- Training: **{len(train)}** baris ({len(train)/len(data):.2%}).",
        f"- Testing: **{len(test)}** baris ({len(test)/len(data):.2%}).",
        f"- Grup unik: **{groups.nunique()}**; grup training: **{len(train_groups)}**; grup testing: **{len(test_groups)}**.",
        f"- Fitur (X): {', '.join(f'`{c}`' for c in FEATURES)}.",
        f"- Target (y): `{TARGET}`; tidak disertakan di X.",
        f"- Grup duplikat identik yang muncul lebih dari sekali: **{int(data.duplicated(GROUP_COLUMNS, keep=False).sum())} baris dalam {int(data.duplicated(GROUP_COLUMNS, keep=False).sum() - data.duplicated(GROUP_COLUMNS).sum())} grup**.",
        f"- Grup identik di kedua partisi: **{len(overlap)}** (verifikasi lulus).", "",
        "## Artefak", "",
        "- `data/processed/splits/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.",
        "- `outputs/figures/splits/training_testing_counts.png` dan `target_distribution_train_test.png` (300 DPI).",
        "- Tiga scatter plot points-only untuk masing-masing partisi berada di `outputs/figures/train/` dan `outputs/figures/test/` (300 DPI).",
        "- Data sumber cleaned tidak diubah. Tidak dilakukan normalisasi, training, atau evaluasi model.", "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    return len(train), len(test)


if __name__ == "__main__":
    n_train, n_test = run_split()
    print(f"Training records: {n_train}")
    print(f"Testing records: {n_test}")
