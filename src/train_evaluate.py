"""Train and evaluate XGBoost using the existing normalized splits."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
SPLITS=ROOT/"data"/"processed"/"splits"
MODELS=ROOT/"models"
TABLES=ROOT/"outputs"/"tables"
FIGURES=ROOT/"outputs"/"figures"/"model_evaluation"
REPORT=ROOT/"outputs"/"reports"/"model_evaluation.md"
FEATURES=["Soil Moisture","Temperature","Usia Tanam"]
TARGET="Volume Penyiraman"
PARAMETERS={"n_estimators":250,"max_depth":2,"learning_rate":0.03,"subsample":0.8,
            "colsample_bytree":0.8,"min_child_weight":3,"reg_lambda":5,
            "objective":"reg:squarederror","random_state":42}


def _metrics(actual, predicted):
    return {"MAE":float(mean_absolute_error(actual,predicted)),
            "RMSE":float(np.sqrt(mean_squared_error(actual,predicted))),
            "R2":float(r2_score(actual,predicted))}


def _cv_summary(model, X, y, groups, splitter):
    scores=cross_validate(model,X,y,groups=groups,cv=splitter,
                          scoring={"mae":"neg_mean_absolute_error","rmse":"neg_root_mean_squared_error","r2":"r2"},
                          return_train_score=False,n_jobs=1)
    results={"MAE":-scores["test_mae"],"RMSE":-scores["test_rmse"],"R2":scores["test_r2"]}
    return {metric:{"folds":[float(x) for x in vals],"mean":float(np.mean(vals)),"std":float(np.std(vals,ddof=1))}
            for metric,vals in results.items()}


def _make_charts(actual,predicted,dummy_pred,model):
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.titlesize":13,"axes.labelsize":11,
                         "figure.facecolor":"white","axes.facecolor":"#FBFCFE"})
    # Actual versus predicted: dots plus the requested ideal diagonal reference.
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(actual,predicted,s=38,alpha=.78,color="#2673A6",edgecolors="white",linewidths=.5,label="Prediksi")
    low=float(min(np.min(actual),np.min(predicted))); high=float(max(np.max(actual),np.max(predicted)))
    ax.plot([low,high],[low,high],linestyle="--",color="#D1495B",linewidth=1.6,label="Garis ideal")
    ax.set_title("Nilai Aktual vs Prediksi XGBoost",weight="semibold",pad=12)
    ax.set_xlabel("Volume Penyiraman Aktual (mL)"); ax.set_ylabel("Volume Penyiraman Prediksi (mL)")
    ax.grid(True,linestyle="--",alpha=.4); ax.set_axisbelow(True); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES/"actual_vs_predicted.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    residual=actual-predicted
    fig,ax=plt.subplots(figsize=(7,5.2))
    ax.scatter(predicted,residual,s=38,alpha=.78,color="#2673A6",edgecolors="white",linewidths=.5)
    ax.axhline(0,color="#D1495B",linestyle="--",linewidth=1.5)
    ax.set_title("Plot Residual XGBoost",weight="semibold",pad=12)
    ax.set_xlabel("Volume Penyiraman Prediksi (mL)"); ax.set_ylabel("Residual (aktual - prediksi, mL)")
    ax.grid(True,linestyle="--",alpha=.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIGURES/"residual_plot.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5.2))
    ax.hist(residual,bins=12,color="#2673A6",edgecolor="white",alpha=.9)
    ax.set_title("Distribusi Residual XGBoost",weight="semibold",pad=12)
    ax.set_xlabel("Residual (aktual - prediksi, mL)"); ax.set_ylabel("Frekuensi")
    ax.grid(axis="y",linestyle="--",alpha=.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIGURES/"residual_distribution.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    importance=pd.Series(model.feature_importances_,index=FEATURES).sort_values()
    fig,ax=plt.subplots(figsize=(7,4.8))
    ax.barh([{"Soil Moisture":"Kelembapan Tanah","Temperature":"Suhu","Usia Tanam":"Usia Tanam"}[n] for n in importance.index],importance.values,color="#2673A6")
    ax.set_title("Kepentingan Fitur XGBoost",weight="semibold",pad=12)
    ax.set_xlabel("Feature importance (gain-based score)"); ax.set_ylabel("Fitur")
    ax.grid(axis="x",linestyle="--",alpha=.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIGURES/"feature_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    xgb=_metrics(actual,predicted); dummy=_metrics(actual,dummy_pred)
    labels=["MAE (mL)","RMSE (mL)","R?"]
    metrics=["MAE","RMSE","R2"]
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,label,metric in zip(axes,labels,metrics):
        values=[xgb[metric],dummy[metric]]
        bars=ax.bar(["XGBoost","Dummy"],values,color=["#2673A6","#E39C37"],width=.62)
        ax.set_title(label,weight="semibold")
        ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=9)
        ax.grid(axis="y",linestyle="--",alpha=.38); ax.set_axisbelow(True)
        if metric=="R2":
            ymin=min(0.0,min(values)-.1); ymax=max(values)+.12
            ax.set_ylim(ymin,ymax)
    fig.suptitle("Perbandingan Kinerja pada Data Testing",fontsize=14,weight="semibold")
    fig.tight_layout()
    fig.savefig(FIGURES/"xgboost_vs_dummy.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    return {"xgb":xgb,"dummy":dummy}


def run_training():
    x_train=pd.read_csv(SPLITS/"X_train_normalized.csv")
    x_test=pd.read_csv(SPLITS/"X_test_normalized.csv")
    y_train=pd.read_csv(SPLITS/"y_train.csv")[TARGET]
    y_test=pd.read_csv(SPLITS/"y_test.csv")[TARGET]
    if list(x_train.columns)!=FEATURES or list(x_test.columns)!=FEATURES:
        raise ValueError("X files must contain only the three requested input features in order")
    if TARGET in x_train.columns or TARGET in x_test.columns:
        raise ValueError("Target leakage: target is present in X")
    if len(x_train)!=len(y_train) or len(x_test)!=len(y_test):
        raise ValueError("Feature-target row alignment check failed")
    if len(x_train) + len(x_test) != 152:
        raise ValueError(f"Unexpected total split size: train={len(x_train)}, test={len(x_test)}")

    # Group duplicates by the complete feature-target record, using training data only.
    complete_train=x_train.copy()
    complete_train[TARGET]=y_train.to_numpy()
    groups=complete_train.groupby(FEATURES+[TARGET],sort=False,dropna=False).ngroup()
    cv=GroupKFold(n_splits=5)
    base_model=XGBRegressor(**PARAMETERS,n_jobs=1,verbosity=0)
    dummy_model=DummyRegressor(strategy="mean")
    xgb_cv=_cv_summary(base_model,x_train,y_train,groups,cv)
    dummy_cv=_cv_summary(dummy_model,x_train,y_train,groups,cv)

    # Final model fit uses all training rows; test set is only evaluated below.
    model=XGBRegressor(**PARAMETERS,n_jobs=1,verbosity=0)
    model.fit(x_train,y_train)
    predictions=model.predict(x_test)
    dummy_final=DummyRegressor(strategy="mean").fit(x_train,y_train)
    dummy_predictions=dummy_final.predict(x_test)
    xgb_metrics=_metrics(y_test,predictions)
    dummy_metrics=_metrics(y_test,dummy_predictions)

    MODELS.mkdir(parents=True,exist_ok=True); TABLES.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODELS/"xgboost_model.joblib")
    pd.DataFrame({"Actual":y_test.to_numpy(),"Predicted_XGBoost":predictions,
                  "Residual_Actual_Minus_Predicted":y_test.to_numpy()-predictions,
                  "Predicted_Dummy_Mean":dummy_predictions}).to_csv(TABLES/"test_predictions.csv",index=False,encoding="utf-8-sig")
    metrics_rows=[]
    for model_name,metrics in [("XGBoost",xgb_metrics),("DummyRegressor",dummy_metrics)]:
        metrics_rows.append({"Model":model_name,"Evaluation":"test",**metrics})
    for model_name,summary in [("XGBoost",xgb_cv),("DummyRegressor",dummy_cv)]:
        for metric,values in summary.items():
            metrics_rows.append({"Model":model_name,"Evaluation":"5-fold GroupKFold train CV","Metric":metric,
                                 "Mean":values["mean"],"Std":values["std"]})
    pd.DataFrame(metrics_rows).to_csv(TABLES/"model_metrics.csv",index=False,encoding="utf-8-sig")
    _make_charts(y_test.to_numpy(),predictions,dummy_predictions,model)

    param_str=", ".join(f"`{k}={v}`" for k,v in PARAMETERS.items())
    lines=["# Laporan Evaluasi Model XGBoost", "",
           "## Prosedur", "",
           "- Input: split normalized yang sudah tersedia; scaler tidak di-fit ulang.",
           "- Fitur: Soil Moisture, Temperature, Usia Tanam. Target: Volume Penyiraman (mL), tidak dinormalisasi.",
           f"- Pemeriksaan alignment: {len(x_train)} pasangan training dan {len(x_test)} pasangan testing; jumlah baris cocok; target tidak ada di X.",
           "- Pembagian data menggunakan GroupShuffleSplit dengan sasaran 70:30; grup duplikat identik dijaga tetap pada satu split.",
           f"- GroupKFold 5-fold memakai training saja; grup identik berdasarkan seluruh tiga fitur dan target. Jumlah grup training: {groups.nunique()}.",
           f"- Model final di-fit pada seluruh {len(x_train)} baris training; test digunakan sekali untuk evaluasi.",
           "- Tidak ada clipping; prediksi mentah disimpan.", "",
           "## Parameter XGBRegressor", "", param_str+".", "`n_jobs=1` digunakan untuk eksekusi deterministik.", "",
           "## Cross-validation training", "", "Ringkasan mean ? standar deviasi antar-fold:", "",
           "| Model | MAE (mL) | RMSE (mL) | R? |", "|---|---:|---:|---:|"]
    for name,summary in [("XGBoost",xgb_cv),("DummyRegressor (mean)",dummy_cv)]:
        lines.append(f"| {name} | {summary['MAE']['mean']:.3f} ? {summary['MAE']['std']:.3f} | {summary['RMSE']['mean']:.3f} ? {summary['RMSE']['std']:.3f} | {summary['R2']['mean']:.3f} ? {summary['R2']['std']:.3f} |")
    lines += ["", "## Evaluasi testing", "", "| Model | MAE (mL) | RMSE (mL) | R? |", "|---|---:|---:|---:|"]
    lines.append(f"| XGBoost | {xgb_metrics['MAE']:.3f} | {xgb_metrics['RMSE']:.3f} | {xgb_metrics['R2']:.3f} |")
    lines.append(f"| DummyRegressor (mean) | {dummy_metrics['MAE']:.3f} | {dummy_metrics['RMSE']:.3f} | {dummy_metrics['R2']:.3f} |")
    if xgb_metrics['MAE'] < dummy_metrics['MAE']:
        interp="Pada data testing, rata-rata kesalahan absolut XGBoost lebih rendah daripada baseline mean."
    else:
        interp="Pada data testing, MAE XGBoost tidak lebih rendah daripada baseline mean."
    if xgb_metrics['R2'] >= 0:
        interp += " R? non-negatif menunjukkan model menjelaskan sebagian variasi target dibanding prediksi konstan berdasarkan rerata testing."
    else:
        interp += " R? negatif menunjukkan model lebih buruk daripada baseline rerata target testing menurut metrik ini."
    lines += ["", "## Interpretasi", "", interp,
              "Semua metrik dan prediksi menggunakan satuan asli Volume Penyiraman (mL). Residual didefinisikan sebagai aktual dikurangi prediksi.",
              "", "## Output", "", "- Model: `models/xgboost_model.joblib`.",
              "- Prediksi testing: `outputs/tables/test_predictions.csv`.",
              "- Metrik: `outputs/tables/model_metrics.csv`.",
              "- Grafik 300 DPI: `outputs/figures/model_evaluation/`.", ""]
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    return {"train_rows":len(x_train),"test_rows":len(x_test),"xgb":xgb_metrics,"dummy":dummy_metrics,
            "xgb_cv":xgb_cv,"dummy_cv":dummy_cv}


if __name__=="__main__":
    result=run_training()
    print(json.dumps(result,indent=2))
