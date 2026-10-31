"""Fit StandardScaler on training features and export normalized splits."""
from __future__ import annotations

from pathlib import Path
import hashlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
FIGURES = ROOT / "outputs" / "figures" / "normalized"
REPORT = ROOT / "outputs" / "reports" / "normalization_report.md"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
WORKBOOK_PATH = ROOT / "data" / "processed" / "normalized_data_formatted.xlsx"
FEATURES = ["Soil Moisture", "Temperature", "Usia Tanam"]
TARGET = "Volume Penyiraman"
LABELS = {
    "Soil Moisture": "Kelembapan Tanah",
    "Temperature": "Suhu",
    "Usia Tanam": "Usia Tanam",
}


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_distributions(before: pd.DataFrame, after: pd.DataFrame) -> list[Path]:
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.titlesize":12,"axes.labelsize":10,
                         "figure.facecolor":"white","axes.facecolor":"#FBFCFE"})
    outputs=[]
    for frame,title,filename,normalized in [
        (before,"Distribusi Fitur Sebelum Normalisasi","feature_distributions_before.png",False),
        (after,"Distribusi Fitur Setelah Normalisasi","feature_distributions_after.png",True),
    ]:
        fig,axes=plt.subplots(1,3,figsize=(13,4.3))
        for ax,col in zip(axes,FEATURES):
            ax.hist(frame[col],bins=18,color="#2673A6",edgecolor="white",linewidth=.6,alpha=.9)
            ax.set_title(LABELS[col],weight="semibold")
            ax.set_xlabel("Nilai ternormalisasi (z-score)" if normalized else LABELS[col])
            ax.set_ylabel("Frekuensi")
            ax.grid(axis="y",linestyle="--",alpha=.38)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle(title,fontsize=15,weight="semibold",y=1.04)
        fig.tight_layout()
        path=FIGURES/filename
        fig.savefig(path,dpi=300,bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def plot_scatter(frame: pd.DataFrame, y: pd.Series, partition: str) -> list[Path]:
    outputs=[]
    for col in FEATURES:
        fig,ax=plt.subplots(figsize=(8,5.2))
        ax.scatter(frame[col],y,s=34,alpha=.76,color="#2673A6",edgecolors="white",linewidths=.45)
        ax.set_title(f"{LABELS[col]} Ternormalisasi terhadap Volume Penyiraman - Data {partition}",pad=12,weight="semibold")
        ax.set_xlabel(f"{LABELS[col]} ternormalisasi (z-score)")
        ax.set_ylabel("Volume Penyiraman (mL)")
        ax.grid(True,linestyle="--",linewidth=.65,alpha=.48,color="#9AA9B8")
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        ax.text(.99,.02,f"n = {len(frame)}",transform=ax.transAxes,ha="right",va="bottom",fontsize=9,color="#526273")
        fig.tight_layout()
        path=FIGURES/f"{partition.lower()}_{ {'Soil Moisture':'soil_moisture','Temperature':'temperature','Usia Tanam':'plant_age'}[col] }_vs_volume.png"
        fig.savefig(path,dpi=300,bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def add_data_sheet(wb: Workbook, name: str, data: pd.DataFrame) -> None:
    ws=wb.create_sheet(name)
    ws.append(list(data.columns))
    for row in data.itertuples(index=False,name=None):
        ws.append(list(row))
    last_col=get_column_letter(len(data.columns))
    table=Table(displayName=name.replace(" ","")+"Table",ref=f"A1:{last_col}{len(data)+1}")
    table.tableStyleInfo=TableStyleInfo(name="TableStyleMedium2",showFirstColumn=False,showLastColumn=False,showRowStripes=True,showColumnStripes=False)
    ws.add_table(table)
    for cell in ws[1]:
        cell.fill=PatternFill("solid",fgColor="17365D")
        cell.font=Font(name="Aptos",size=10,bold=True,color="FFFFFF")
        cell.alignment=Alignment(horizontal="center",vertical="center",wrap_text=True)
        cell.border=Border(bottom=Side(style="medium",color="4472C4"))
    ws.row_dimensions[1].height=28
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font=Font(name="Aptos",size=10,color="243247")
            cell.alignment=Alignment(horizontal="right" if isinstance(cell.value,(int,float)) else "left",vertical="center")
            cell.number_format="0.0000" if isinstance(cell.value,(int,float)) else "General"
            cell.border=Border(bottom=Side(style="thin",color="D9E2F3"))
    for i,col in enumerate(data.columns,1):
        width=min(max(len(str(col))+3,16),30)
        ws.column_dimensions[get_column_letter(i)].width=width
    ws.freeze_panes="A2"
    ws.sheet_view.showGridLines=False


def run_normalization() -> dict[str,int]:
    x_train_path=SPLITS/"X_train.csv"
    x_test_path=SPLITS/"X_test.csv"
    y_train_path=SPLITS/"y_train.csv"
    y_test_path=SPLITS/"y_test.csv"
    y_hashes={p.name:sha256(p) for p in (y_train_path,y_test_path)}
    x_train=pd.read_csv(x_train_path)
    x_test=pd.read_csv(x_test_path)
    y_train=pd.read_csv(y_train_path)
    y_test=pd.read_csv(y_test_path)
    if list(x_train.columns)!=FEATURES or list(x_test.columns)!=FEATURES:
        raise ValueError("Split feature columns do not match the expected three features")
    if TARGET in x_train.columns or TARGET in x_test.columns:
        raise ValueError("Target leakage: target found in an X file")
    if list(y_train.columns)!=[TARGET] or list(y_test.columns)!=[TARGET]:
        raise ValueError("Target split columns do not match expected target")
    if len(x_train)!=len(y_train) or len(x_test)!=len(y_test):
        raise ValueError("Feature and target partition lengths do not match")

    scaler=StandardScaler()
    train_values=scaler.fit_transform(x_train[FEATURES])
    test_values=scaler.transform(x_test[FEATURES])
    x_train_normalized=pd.DataFrame(train_values,columns=FEATURES)
    x_test_normalized=pd.DataFrame(test_values,columns=FEATURES)
    train_mean=x_train_normalized.mean()
    train_std=x_train_normalized.std(ddof=0)
    if not np.allclose(train_mean.to_numpy(),0,atol=1e-12) or not np.allclose(train_std.to_numpy(),1,atol=1e-12):
        raise AssertionError("StandardScaler verification failed on training data")

    FIGURES.mkdir(parents=True,exist_ok=True)
    (ROOT/"models").mkdir(parents=True,exist_ok=True)
    x_train_normalized.to_csv(SPLITS/"X_train_normalized.csv",index=False,encoding="utf-8-sig")
    x_test_normalized.to_csv(SPLITS/"X_test_normalized.csv",index=False,encoding="utf-8-sig")
    joblib.dump(scaler,SCALER_PATH)
    distribution_paths=plot_distributions(x_train,x_train_normalized)
    scatter_paths=plot_scatter(x_train_normalized,y_train[TARGET],"Training")+plot_scatter(x_test_normalized,y_test[TARGET],"Testing")

    wb=Workbook()
    default=wb.active
    wb.remove(default)
    add_data_sheet(wb,"X Train Normalized",x_train_normalized)
    add_data_sheet(wb,"X Test Normalized",x_test_normalized)
    add_data_sheet(wb,"y Train",y_train)
    add_data_sheet(wb,"y Test",y_test)
    charts=wb.create_sheet("Charts")
    charts.sheet_view.showGridLines=False
    charts["A1"]="Visualisasi Normalisasi dan Scatter - Data Aktual"
    charts["A1"].font=Font(name="Aptos Display",size=16,bold=True,color="17365D")
    charts.column_dimensions["A"].width=18
    for i,path in enumerate(distribution_paths+scatter_paths):
        image=XLImage(str(path))
        image.width=760
        image.height=440
        image.anchor=f"A{3+i*24}"
        charts.add_image(image)
    wb.save(WORKBOOK_PATH)

    # Ensure target files were not touched by this stage.
    if any(sha256(path)!=y_hashes[path.name] for path in (y_train_path,y_test_path)):
        raise AssertionError("Target split file changed unexpectedly")
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    lines=["# Laporan Normalisasi", "",
           "**Metode:** `StandardScaler`; scaler di-fit hanya pada `X_train` lalu digunakan untuk mentransformasi `X_test`.",
           "**Fitur:** Soil Moisture, Temperature, Usia Tanam. Target Volume Penyiraman tidak dinormalisasi.", "",
           "## Ukuran split", "",f"- Training: **{len(x_train)}** baris.",f"- Testing: **{len(x_test)}** baris.", "",
           "## Verifikasi statistik training", "", "Rata-rata dan standar deviasi populasi (ddof=0) setelah normalisasi:", "",
           "| Fitur | Mean | Std deviasi |", "|---|---:|---:|"]
    for feature in FEATURES:
        lines.append(f"| {feature} | {train_mean[feature]:.12f} | {train_std[feature]:.12f} |")
    lines += ["", "Mean mendekati 0 dan standar deviasi mendekati 1 untuk semua fitur training.",
              "", "## Artefak", "",
              "- `data/processed/splits/X_train_normalized.csv` dan `X_test_normalized.csv`.",
              "- `models/scaler.joblib`.",
              "- `data/processed/normalized_data_formatted.xlsx` dengan worksheet data X/y dan `Charts`.",
              "- `outputs/figures/normalized/feature_distributions_before.png` dan `feature_distributions_after.png`.",
              "- Scatter points-only training/testing: `outputs/figures/normalized/` (masing-masing tiga grafik, 300 DPI).",
              "- `y_train.csv`, `y_test.csv`, split asli, dan cleaned dataset tidak diubah.",
              "- Tidak dilakukan pelatihan model XGBoost.", ""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    return {"train":len(x_train),"test":len(x_test)}


if __name__=="__main__":
    counts=run_normalization()
    print(f"Training records: {counts['train']}")
    print(f"Testing records: {counts['test']}")
