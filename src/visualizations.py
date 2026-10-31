"""Scatter plots for raw, filtered, and cleaned irrigation datasets."""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_ROOT = PROJECT_ROOT / "outputs" / "figures"
PLOTS = (
    ("Soil Moisture", "Kelembapan Tanah (%)", "soil_moisture_vs_volume.png"),
    ("Temperature", "Suhu (derajat C)", "temperature_vs_volume.png"),
    ("Usia Tanam", "Usia Tanam (hari)", "plant_age_vs_volume.png"),
)
TARGET = "Volume Penyiraman"


def _numeric(series: pd.Series) -> pd.Series:
    """Parse numeric values for plotting only; source data are never modified."""
    text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _safe_stage(stage: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "_", stage.strip().lower()).strip("_")
    if not value:
        raise ValueError("stage must contain letters or numbers")
    return value


def generate_scatter_plots(data: pd.DataFrame, stage: str) -> list[Path]:
    """Create three 300 DPI scatter plots using values in *data*.

    Numeric conversion is temporary and limited to plotted columns. Rows with
    unparseable X/Y values cannot be placed on a scatter plot and are omitted
    from that plot, while the input DataFrame remains unchanged.
    """
    required = [column for column, _, _ in PLOTS] + [TARGET]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {', '.join(missing)}")
    stage_dir = FIGURES_ROOT / _safe_stage(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    y_values = _numeric(data[TARGET])
    written: list[Path] = []
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "#FBFCFE",
    })
    for column, x_label, filename in PLOTS:
        x_values = _numeric(data[column])
        valid = x_values.notna() & y_values.notna()
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        ax.scatter(x_values[valid], y_values[valid], s=34, alpha=0.76,
                   color="#2673A6", edgecolors="white", linewidths=0.45)
        ax.set_title(f"{x_label.split(' (')[0]} terhadap Volume Penyiraman - {stage.title()}", pad=12, weight="semibold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Volume Penyiraman (mL)")
        ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.5, color="#9AA9B8")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.99, 0.02, f"n = {int(valid.sum())}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#526273")
        fig.tight_layout()
        destination = stage_dir / filename
        invalid = set('<>:"/\\|?*')
        if any(character in invalid for character in destination.name):
            plt.close(fig)
            raise ValueError(f"Invalid Windows filename: {destination.name!r}")
        temporary = stage_dir / f".{destination.stem}.tmp.png"
        try:
            fig.savefig(temporary, dpi=300, bbox_inches="tight", format="png")
            temporary.replace(destination)
        finally:
            plt.close(fig)
            if temporary.exists():
                temporary.unlink()
        written.append(destination)
    return written


def embed_scatter_plots(workbook_path: str | Path, plot_paths: list[str | Path]) -> Path:
    """Embed plot PNGs on a Charts worksheet in an existing formatted workbook."""
    path = Path(workbook_path)
    wb = load_workbook(path)
    if "Charts" in wb.sheetnames:
        del wb["Charts"]
    ws = wb.create_sheet("Charts")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Grafik Sebaran - Dataset Tersaring"
    ws["A1"].font = __import__("openpyxl").styles.Font(name="Aptos Display", size=16, bold=True, color="17365D")
    ws.column_dimensions["A"].width = 18
    for index, plot_path in enumerate(plot_paths):
        image = XLImage(str(plot_path))
        image.width = 820
        image.height = 540
        image.anchor = f"A{3 + index * 30}"
        ws.add_image(image)
    wb.save(path)
    return path


def generate_after_processing(data: pd.DataFrame, stage: str,
                              formatted_workbook: str | Path | None = None) -> list[Path]:
    """Post-filtering/cleaning hook: generate stage plots and embed when requested."""
    plot_paths = generate_scatter_plots(data, stage)
    if formatted_workbook is not None:
        embed_scatter_plots(formatted_workbook, plot_paths)
    return plot_paths


def generate_available_stages() -> dict[str, list[Path]]:
    """Render available stages and embed charts into available formatted workbooks."""
    stages: list[tuple[str, pd.DataFrame, Path | None]] = []
    raw_path = PROJECT_ROOT / "data" / "raw" / "data_mentah.xlsx"
    if raw_path.exists():
        stages.append(("raw", pd.read_excel(raw_path), None))
    processed = PROJECT_ROOT / "data" / "processed"
    for stage in ("filtered", "cleaned"):
        csv_path = processed / f"data_{stage}.csv"
        if csv_path.exists():
            formatted = processed / f"data_{stage}_formatted.xlsx"
            stages.append((stage, pd.read_csv(csv_path), formatted if formatted.exists() else None))
    result: dict[str, list[Path]] = {}
    for stage, data, workbook in stages:
        result[stage] = generate_after_processing(data, stage, workbook)
    return result


if __name__ == "__main__":
    outputs = generate_available_stages()
    for stage, paths in outputs.items():
        print(f"{stage}: {len(paths)} plots")
        for path in paths:
            print(f"  {path.relative_to(PROJECT_ROOT)}")
