"""Export aligned actual/predicted tables and plots for the current data split."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures" / "model_evaluation"
FEATURES = ["Soil Moisture", "Temperature", "Usia Tanam"]
TARGET = "Volume Penyiraman"
HEADERS = [
    "Nomor Sampel", "Kelembapan Tanah", "Suhu", "Usia Tanam",
    "Volume Penyiraman Aktual (mL)", "Volume Penyiraman Prediksi (mL)", "Jenis Dataset",
]


def main() -> None:
    model = joblib.load(ROOT / "models" / "xgboost_model.joblib")
    scaler = joblib.load(ROOT / "models" / "scaler.joblib")
    saved_test = pd.read_csv(TABLES / "test_predictions.csv")
    sections: list[pd.DataFrame] = []
    generated_test: np.ndarray | None = None

    for split, label in (("train", "Training"), ("test", "Testing")):
        X = pd.read_csv(SPLITS / f"X_{split}.csv")
        y = pd.read_csv(SPLITS / f"y_{split}.csv")[TARGET].to_numpy()
        if list(X.columns) != FEATURES or len(X) != len(y):
            raise ValueError(f"Invalid feature schema or row alignment for {split}")
        scaled = pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES)
        predictions = np.asarray(model.predict(scaled), dtype=float)
        if split == "test":
            if len(saved_test) != len(predictions):
                raise ValueError("Saved test prediction count does not match the current test split")
            generated_test = predictions
            if not np.allclose(predictions, saved_test["Predicted_XGBoost"], rtol=0, atol=3e-5):
                raise ValueError("Model/scaler predictions do not reproduce saved test predictions")
            # Keep precisely the predictions published by the training stage.
            predictions = saved_test["Predicted_XGBoost"].to_numpy(dtype=float)
        sections.append(pd.DataFrame({
            "Nomor Sampel": np.arange(1, len(X) + 1),
            "Kelembapan Tanah": X[FEATURES[0]].to_numpy(),
            "Suhu": X[FEATURES[1]].to_numpy(),
            "Usia Tanam": X[FEATURES[2]].to_numpy(),
            "Volume Penyiraman Aktual (mL)": y,
            "Volume Penyiraman Prediksi (mL)": predictions,
            "Jenis Dataset": label,
        }))

    complete = pd.concat(sections, ignore_index=True)
    if list(complete.columns) != HEADERS:
        raise AssertionError("Unexpected comparison table schema")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    complete.to_csv(TABLES / "complete_actual_vs_predicted.csv", index=False,
                    encoding="utf-8-sig", float_format="%.10g")

    # Preserve the established testing-only comparison table as well.
    testing = complete[complete["Jenis Dataset"].eq("Testing")].reset_index(drop=True)
    actual = testing["Volume Penyiraman Aktual (mL)"].to_numpy()
    predicted = testing["Volume Penyiraman Prediksi (mL)"].to_numpy()
    legacy = pd.DataFrame({
        "Nomor Sampel": np.arange(1, len(testing) + 1),
        "Aktual (mL)": actual,
        "Prediksi (mL)": predicted,
        "Error Absolut (mL)": np.abs(actual - predicted),
    })
    legacy.to_csv(TABLES / "actual_vs_predicted.csv", index=False,
                  encoding="utf-8-sig", float_format="%.10g")

    with pd.ExcelWriter(TABLES / "complete_actual_vs_predicted.xlsx", engine="openpyxl") as writer:
        complete.to_excel(writer, index=False, sheet_name="Perbandingan")
        counts = complete["Jenis Dataset"].value_counts()
        pd.DataFrame({"Jenis Dataset": ["Training", "Testing", "Total"],
                      "Jumlah Record": [int(counts["Training"]), int(counts["Testing"]), len(complete)]}).to_excel(
                          writer, index=False, sheet_name="Ringkasan")
    with pd.ExcelWriter(TABLES / "actual_vs_predicted.xlsx", engine="openpyxl") as writer:
        legacy.to_excel(writer, index=False, sheet_name="Testing")

    navy = PatternFill("solid", fgColor="17365D")
    for filename, sheet, widths, numeric_cols in (
        ("complete_actual_vs_predicted.xlsx", "Perbandingan", [16, 20, 14, 14, 31, 34, 18], range(2, 7)),
        ("actual_vs_predicted.xlsx", "Testing", [16, 22, 22, 24], range(2, 5)),
    ):
        workbook = load_workbook(TABLES / filename)
        worksheet = workbook[sheet]
        worksheet.freeze_panes = "A2"
        worksheet.sheet_view.showGridLines = False
        for cell in worksheet[1]:
            cell.fill = navy
            cell.font = Font(name="Aptos", size=10, bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        worksheet.row_dimensions[1].height = 32
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name="Aptos", size=10, color="1F2937")
                cell.alignment = Alignment(horizontal="center" if cell.column == 1 else "right", vertical="center")
                if cell.column in numeric_cols:
                    cell.number_format = "#,##0.00"
        for i, width in enumerate(widths, 1):
            worksheet.column_dimensions[get_column_letter(i)].width = width
        if sheet == "Perbandingan":
            table = Table(displayName="ActualPredicted70to30", ref=f"A1:G{len(complete) + 1}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True,
                                                   showColumnStripes=False, showFirstColumn=False,
                                                   showLastColumn=False)
            worksheet.add_table(table)
        else:
            worksheet.auto_filter.ref = f"A1:D{len(legacy) + 1}"
        workbook.save(TABLES / filename)

    workbook = load_workbook(TABLES / "complete_actual_vs_predicted.xlsx")
    summary = workbook["Ringkasan"]
    summary.freeze_panes = "A2"
    summary.sheet_view.showGridLines = False
    summary.row_dimensions[1].height = 28
    for cell in summary[1]:
        cell.fill = navy
        cell.font = Font(name="Aptos", size=11, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in summary.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Aptos", size=11, color="1F2937")
            cell.alignment = Alignment(vertical="center")
            if cell.row % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="EAF2F8")
            if cell.column == 2:
                cell.number_format = "#,##0"
    summary.column_dimensions["A"].width = 24
    summary.column_dimensions["B"].width = 20
    workbook.save(TABLES / "complete_actual_vs_predicted.xlsx")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "figure.facecolor": "white",
                         "axes.facecolor": "#FBFCFE"})
    for label, color in (("Training", "#2673A6"), ("Testing", "#E39C37")):
        part = complete[complete["Jenis Dataset"].eq(label)]
        x = part["Nomor Sampel"].to_numpy()
        y_actual = part["Volume Penyiraman Aktual (mL)"].to_numpy()
        y_pred = part["Volume Penyiraman Prediksi (mL)"].to_numpy()
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(x, y_actual, color="#17365D", marker="o", markersize=4, linewidth=1.8, label="Aktual")
        ax.plot(x, y_pred, color=color, marker="s", markersize=4, linewidth=1.6,
                linestyle="--", label="Prediksi")
        ax.set(title=f"Volume Penyiraman Aktual vs Prediksi per Sampel — {label} (n={len(part)})",
               xlabel="Nomor Sampel", ylabel="Volume Penyiraman (mL)")
        ax.set_xticks(x if len(x) <= 40 else x[::5])
        ax.grid(True, linestyle="--", alpha=.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"actual_vs_predicted_{label.lower()}_by_sample.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        lo = float(min(y_actual.min(), y_pred.min()))
        hi = float(max(y_actual.max(), y_pred.max()))
        pad = (hi - lo) * .04
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        ax.scatter(y_actual, y_pred, s=45, alpha=.8, color=color,
                   edgecolors="white", linewidths=.6, label=f"{label} (n={len(part)})")
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#D1495B", lw=1.6,
                label="Prediksi ideal")
        ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad),
               title=f"Aktual vs Prediksi — {label}",
               xlabel="Volume Penyiraman Aktual (mL)", ylabel="Volume Penyiraman Prediksi (mL)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"actual_vs_predicted_{label.lower()}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    report = [
        "# Perbandingan Aktual dan Prediksi", "",
        f"Dataset memakai split sasaran training:testing **70:30**. Group split menghasilkan **{len(sections[0])}** baris training dan **{len(testing)}** testing dari {len(complete)} record.", "",
        f"Pada data testing, MAE = **{mae:.3f} mL**, RMSE = **{rmse:.3f} mL**, dan R-squared = **{r2:.3f}**.", "",
        "Metrik dihitung dari prediksi XGBoost testing yang tersimpan dan disejajarkan dengan target `y_test`.", "",
        "Grafik aktual dan prediksi per sampel tersedia pada `outputs/figures/model_evaluation/actual_vs_predicted_training_by_sample.png` dan `actual_vs_predicted_testing_by_sample.png`. Scatter aktual vs prediksi tersedia pada file `actual_vs_predicted_training.png` dan `actual_vs_predicted_testing.png`.", "",
        "Tabel gabungan: `outputs/tables/complete_actual_vs_predicted.csv` dan `.xlsx`; workbook memuat ringkasan jumlah record. Tabel testing: `outputs/tables/actual_vs_predicted.csv` dan `.xlsx`.", "",
    ]
    (ROOT / "outputs" / "reports" / "actual_vs_predicted.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Rows: training={len(sections[0])}, testing={len(testing)}, total={len(complete)}")
    print(f"Testing metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}")
    print(f"Maximum test prediction replay difference: {np.max(np.abs(generated_test - saved_test['Predicted_XGBoost'].to_numpy())):.8g}")


if __name__ == "__main__":
    main()
