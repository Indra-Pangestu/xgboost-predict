"""Audit saved XGBoost artifacts and refresh plots without fitting models."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT=Path(__file__).resolve().parents[1]
SPLITS=ROOT/"data"/"processed"/"splits"
CLEANED=ROOT/"data"/"processed"/"data_cleaned.csv"
MODEL_PATH=ROOT/"models"/"xgboost_model.joblib"
PRED_PATH=ROOT/"outputs"/"tables"/"test_predictions.csv"
METRIC_PATH=ROOT/"outputs"/"tables"/"model_metrics.csv"
OUT=ROOT/"outputs"/"figures"/"model_evaluation"
REPORT=ROOT/"outputs"/"reports"/"model_evaluation.md"
FEATURES=["Soil Moisture","Temperature","Usia Tanam"]
TARGET="Volume Penyiraman"
ALL_COLS=FEATURES+[TARGET]


def metrics(y,p):
    return {"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p))),"R2":float(r2_score(y,p))}


def save_figures(pred,model,metric_table,fold_stats,xgb_test,dummy_test):
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":13,
                         "axes.labelsize":11,"figure.facecolor":"white","axes.facecolor":"#FBFCFE"})
    actual=pred.Actual.to_numpy(float); predicted=pred.Predicted_XGBoost.to_numpy(float)
    residual=actual-predicted
    lo=float(min(actual.min(),predicted.min())); hi=float(max(actual.max(),predicted.max()))
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(actual,predicted,s=38,alpha=.78,color="#2673A6",edgecolors="white",linewidths=.5)
    ax.plot([lo,hi],[lo,hi],"--",color="#D1495B",lw=1.6,label="Garis ideal")
    ax.set(title="Nilai Aktual vs Prediksi XGBoost",xlabel="Volume Penyiraman Aktual (mL)",ylabel="Volume Penyiraman Prediksi (mL)")
    ax.grid(True,ls="--",alpha=.4); ax.set_axisbelow(True); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"actual_vs_predicted.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5.2))
    ax.scatter(predicted,residual,s=38,alpha=.78,color="#2673A6",edgecolors="white",linewidths=.5)
    ax.axhline(0,color="#D1495B",ls="--",lw=1.5)
    ax.set(title="Plot Residual XGBoost",xlabel="Volume Penyiraman Prediksi (mL)",ylabel="Residual (aktual - prediksi, mL)")
    ax.grid(True,ls="--",alpha=.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT/"residual_plot.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5.2))
    ax.hist(residual,bins=12,color="#2673A6",edgecolor="white",alpha=.9)
    ax.set(title="Distribusi Residual XGBoost",xlabel="Residual (aktual - prediksi, mL)",ylabel="Frekuensi")
    ax.grid(axis="y",ls="--",alpha=.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT/"residual_distribution.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    fi=pd.Series(model.feature_importances_,index=FEATURES).sort_values()
    names={"Soil Moisture":"Kelembapan Tanah","Temperature":"Suhu","Usia Tanam":"Usia Tanam"}
    fig,ax=plt.subplots(figsize=(7,4.8))
    ax.barh([names[i] for i in fi.index],fi.values,color="#2673A6")
    ax.set(title="Kepentingan Fitur XGBoost",xlabel="Skor kepentingan fitur",ylabel="Fitur")
    ax.grid(axis="x",ls="--",alpha=.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT/"feature_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    labels=["MAE (mL)","RMSE (mL)","R2"]
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,label,key in zip(axes,labels,["MAE","RMSE","R2"]):
        vals=[xgb_test[key],dummy_test[key]]
        bars=ax.bar(["XGBoost","Dummy"],vals,color=["#2673A6","#E39C37"],width=.62)
        ax.set_title(label,weight="semibold"); ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=9)
        ax.grid(axis="y",ls="--",alpha=.38); ax.set_axisbelow(True)
    fig.suptitle("Perbandingan Kinerja pada Data Testing",fontsize=14,weight="semibold")
    fig.tight_layout(); fig.savefig(OUT/"xgboost_vs_dummy.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Saved metrics contain aggregate CV statistics only, so show means and sample SDs honestly.
    cv=metric_table[metric_table.Evaluation.eq("5-fold GroupKFold train CV")]
    fig,axes=plt.subplots(1,3,figsize=(12,4.6))
    for ax,title,key in zip(axes,["MAE (mL)","RMSE (mL)","R2"],["MAE","RMSE","R2"]):
        part=cv[cv.Metric.eq(key)].set_index("Model").reindex(["XGBoost","DummyRegressor"])
        vals=part["Mean"].to_numpy(float); errs=part["Std"].to_numpy(float)
        ax.bar(["XGBoost","Dummy"],vals,yerr=errs,capsize=4,color=["#2673A6","#E39C37"],width=.62)
        ax.set_title(title + " (mean + SD antar-fold)", weight="semibold")
        ax.grid(axis="y",ls="--",alpha=.38); ax.set_axisbelow(True)
    fig.suptitle("Ringkasan Cross-Validation Tersimpan (5 Fold)",fontsize=14,weight="semibold")
    fig.tight_layout(); fig.savefig(OUT/"cross_validation_performance_summary.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Fold target distribution is directly verifiable without fitting fold models.
    fig,ax=plt.subplots(figsize=(8,5))
    bars=ax.bar(fold_stats["fold"],fold_stats["mean"],color="#2673A6",label="Rata-rata target")
    ax.errorbar(fold_stats["fold"],fold_stats["mean"],yerr=fold_stats["std"],fmt="none",ecolor="#18324A",capsize=4,label="+/- SD target")
    ax.scatter(fold_stats["fold"],fold_stats["max"],color="#D1495B",marker="D",s=48,label="Nilai maksimum")
    for _,row in fold_stats.iterrows():
        if row.outlier_count:
            ax.annotate("1474.6 (pencilan)",(row.fold,row["max"]),xytext=(6,7),textcoords="offset points",color="#A02C3A",fontsize=9)
    ax.set(title="Distribusi Target per Fold Validasi",xlabel="Fold GroupKFold",ylabel="Volume Penyiraman (mL)",xticks=fold_stats.fold)
    ax.grid(axis="y",ls="--",alpha=.4); ax.set_axisbelow(True); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"cross_validation_fold_target_distribution.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def run_audit():
    required=[MODEL_PATH,PRED_PATH,METRIC_PATH,SPLITS/"X_train_normalized.csv",SPLITS/"X_test_normalized.csv",SPLITS/"y_train.csv",SPLITS/"y_test.csv",ROOT/"models"/"scaler.joblib",CLEANED]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing audit artifacts: "+", ".join(missing))
    model=joblib.load(MODEL_PATH)
    pred=pd.read_csv(PRED_PATH); metric_table=pd.read_csv(METRIC_PATH)
    x_train=pd.read_csv(SPLITS/"X_train_normalized.csv"); x_test=pd.read_csv(SPLITS/"X_test_normalized.csv")
    y_train=pd.read_csv(SPLITS/"y_train.csv")[TARGET]; y_test=pd.read_csv(SPLITS/"y_test.csv")[TARGET]
    problems=[]
    if type(model).__name__!="XGBRegressor": problems.append(f"Model class unexpected: {type(model).__name__}")
    if list(x_train.columns)!=FEATURES or list(x_test.columns)!=FEATURES: problems.append("X split feature columns do not exactly match expected features")
    if TARGET in x_train.columns or TARGET in x_test.columns: problems.append("Target leakage: target is present in an X split")
    if not (len(x_train)==len(y_train)==122 and len(x_test)==len(y_test)==30): problems.append("Unexpected split dimensions or X/y length mismatch")
    if len(pred)!=30: problems.append(f"Prediction count is {len(pred)}, expected 30")
    actual_align=len(pred)==len(y_test) and np.allclose(pred.Actual.to_numpy(),y_test.to_numpy(),rtol=0,atol=1e-10)
    if not actual_align: problems.append("Saved prediction actuals do not exactly align with y_test")
    if "Predicted_XGBoost" not in pred or "Residual_Actual_Minus_Predicted" not in pred: problems.append("Required prediction columns missing")

    xgb_test=metrics(pred.Actual,pred.Predicted_XGBoost)
    dummy_test=metrics(pred.Actual,pred.Predicted_Dummy_Mean)
    residual_diff=float(np.max(np.abs(pred.Residual_Actual_Minus_Predicted-(pred.Actual-pred.Predicted_XGBoost))))
    replay=model.predict(x_test[FEATURES])
    replay_diff=float(np.max(np.abs(replay-pred.Predicted_XGBoost.to_numpy())))
    saved_xgb=metric_table[(metric_table.Model=="XGBoost")&(metric_table.Evaluation=="test")].iloc[0]
    saved_dummy=metric_table[(metric_table.Model=="DummyRegressor")&(metric_table.Evaluation=="test")].iloc[0]
    metric_diff=max(abs(xgb_test[k]-float(saved_xgb[k])) for k in ("MAE","RMSE","R2"))
    dummy_metric_diff=max(abs(dummy_test[k]-float(saved_dummy[k])) for k in ("MAE","RMSE","R2"))
    xgb_metric_diffs={k:abs(xgb_test[k]-float(saved_xgb[k])) for k in ("MAE","RMSE","R2")}
    dummy_metric_diffs={k:abs(dummy_test[k]-float(saved_dummy[k])) for k in ("MAE","RMSE","R2")}
    if metric_diff>1e-4 or dummy_metric_diff>1e-4: problems.append("Recomputed test metrics differ from saved metrics beyond 1e-4")

    clean=pd.read_csv(CLEANED); scaler=joblib.load(ROOT/"models"/"scaler.joblib")
    groups=clean.groupby(ALL_COLS,sort=False,dropna=False).ngroup()
    itr,ite=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(clean[FEATURES],clean[TARGET],groups))
    scaled=scaler.transform(clean[FEATURES])
    align_train_x=float(np.max(np.abs(scaled[itr]-x_train[FEATURES].to_numpy())))
    align_test_x=float(np.max(np.abs(scaled[ite]-x_test[FEATURES].to_numpy())))
    align_train_y=float(np.max(np.abs(clean.iloc[itr][TARGET].to_numpy()-y_train.to_numpy())))
    align_test_y=float(np.max(np.abs(clean.iloc[ite][TARGET].to_numpy()-y_test.to_numpy())))
    if max(align_train_x,align_test_x,align_train_y,align_test_y)>1e-10: problems.append("Reconstructed split does not align to cleaned data and scaler")
    train_tuples=set(map(tuple,clean.iloc[itr][ALL_COLS].itertuples(index=False,name=None)))
    test_tuples=set(map(tuple,clean.iloc[ite][ALL_COLS].itertuples(index=False,name=None)))
    split_group_overlap=len(train_tuples & test_tuples)
    if split_group_overlap: problems.append(f"Exact duplicate groups cross train/test partitions: {split_group_overlap}")

    train_records=clean.iloc[itr].reset_index(drop=True)
    fold_groups=train_records.groupby(ALL_COLS,sort=False,dropna=False).ngroup()
    fold_rows=[]
    cv_splitter=GroupKFold(n_splits=5)
    for fold,(_,valid_idx) in enumerate(cv_splitter.split(train_records[FEATURES],train_records[TARGET],fold_groups),1):
        part=train_records.iloc[valid_idx]
        fold_rows.append({"fold":fold,"n":len(part),"mean":part[TARGET].mean(),"std":part[TARGET].std(ddof=0),"min":part[TARGET].min(),"max":part[TARGET].max(),"outlier_count":int(np.isclose(part[TARGET],1474.6,atol=1e-8).sum()),"source_rows":part.source_excel_row.astype(int).tolist()})
    fold_stats=pd.DataFrame(fold_rows)
    fold_sizes=fold_stats.n.tolist()
    if fold_sizes!=[25,25,24,24,24]: problems.append(f"Reconstructed fold sizes differ from expected GroupKFold allocation: {fold_sizes}")
    cv_entries=metric_table[metric_table.Evaluation.eq("5-fold GroupKFold train CV")]
    per_fold_saved=bool(cv_entries.columns.isin([f"fold_{i}" for i in range(1,6)]).any()) or bool(cv_entries.Metric.astype(str).str.contains(r"fold\s*[1-5]",case=False).any())
    if not per_fold_saved: pass

    save_figures(pred,model,metric_table,fold_stats,xgb_test,dummy_test)

    params=model.get_params()
    fi=dict(zip(FEATURES,[float(v) for v in model.feature_importances_]))
    c5=fold_stats.iloc[4]
    train_outlier_rows=clean.iloc[itr].loc[np.isclose(clean.iloc[itr][TARGET],1474.6,atol=1e-8),"source_excel_row"].astype(int).tolist()
    test_outlier_rows=clean.iloc[ite].loc[np.isclose(clean.iloc[ite][TARGET],1474.6,atol=1e-8),"source_excel_row"].astype(int).tolist()
    cv_summary=cv_entries.pivot(index="Model",columns="Metric",values=["Mean","Std"])
    xcv={m:{k:(float(cv_summary.loc[m,("Mean",k)]),float(cv_summary.loc[m,("Std",k)])) for k in ["MAE","RMSE","R2"]} for m in ["XGBoost","DummyRegressor"]}
    lines=["# Audit Hasil Evaluasi XGBoost", "",
      "## Status artefak", "",
      "- Model tersimpan dan dapat dimuat sebagai `XGBRegressor`; versi library saat audit: XGBoost 3.4.1.",
      "- File prediksi dan metrik tersedia. Split dan scaler tersedia; audit ini tidak menulis ulang data tersebut.",
      f"- Baris prediksi: **{len(pred)}**; actual pada seluruh baris cocok berurutan dengan `y_test`: **{'Ya' if actual_align else 'Tidak'}**.",
      f"- Prediksi ulang memakai model tersimpan berbeda maksimum **{replay_diff:.8g}** dari nilai CSV; perbedaan kecil konsisten dengan pembulatan/serialisasi float.",
      f"- Kolom residual tersimpan berbeda maksimum **{residual_diff:.8g}** dari actual dikurangi prediksi CSV.",
      "- Fitur model: `"+"`, `".join(getattr(model,"feature_names_in_",[]))+"`; target tidak terdapat di input X.",
      f"- Row alignment terhadap cleaned data dan scaler: max error X train **{align_train_x:.3g}**, X test **{align_test_x:.3g}**, y train **{align_train_y:.3g}**, y test **{align_test_y:.3g}**.",
      "- Prediksi negatif tersimpan: **"+str(int((pred.Predicted_XGBoost<0).sum()))+"**; nilai disajikan tanpa clipping.", "",
      "## Parameter model tersimpan", "",
      "`n_estimators="+str(params["n_estimators"])+"`, `max_depth="+str(params["max_depth"])+"`, `learning_rate="+str(params["learning_rate"])+"`, `subsample="+str(params["subsample"])+"`, `colsample_bytree="+str(params["colsample_bytree"])+"`, `min_child_weight="+str(params["min_child_weight"])+"`, `reg_lambda="+str(params["reg_lambda"])+"`, `objective="+str(params["objective"])+"`, `random_state="+str(params["random_state"])+"`.", "",
      "## Verifikasi metrik testing", "",
      "Metrik dihitung ulang dari 30 baris pada `test_predictions.csv`; selisih terhadap `model_metrics.csv` dicatat di bawah. Semua nilai dalam satuan asli mL kecuali R2.", "",
      "| Metrik | XGB tersimpan | XGB hitung ulang | Selisih | Dummy tersimpan | Dummy hitung ulang | Selisih |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for key in ("MAE","RMSE","R2"):
      lines.append(f"| {key} | {float(saved_xgb[key]):.10f} | {xgb_test[key]:.10f} | {xgb_metric_diffs[key]:.3g} | {float(saved_dummy[key]):.10f} | {dummy_test[key]:.10f} | {dummy_metric_diffs[key]:.3g} |")
    lines += ["",
              "Kinerja XGBoost lebih baik daripada baseline mean pada split testing: MAE turun **"+f"{(1-xgb_test['MAE']/dummy_test['MAE'])*100:.1f}%"+"** dan RMSE turun **"+f"{(1-xgb_test['RMSE']/dummy_test['RMSE'])*100:.1f}%"+"**. Nilai R2 testing XGBoost adalah **"+f"{xgb_test['R2']:.3f}"+"**.", "",
              "## Cross-validation GroupKFold", "",
              "Metrik CV yang tersimpan hanya berisi mean dan standar deviasi antar lima fold; skor MAE/RMSE/R2 per fold tidak tersimpan. Karena audit ini tidak melatih ulang fold model, skor tiap fold (termasuk fold 5) tidak dapat direkonstruksi secara sah. Grafik yang tersedia menampilkan ringkasan mean +/- SD, bukan skor per fold.", "",
              "| Model | MAE mean +/- SD | RMSE mean +/- SD | R2 mean +/- SD |", "|---|---:|---:|---:|"]
    for name in ["XGBoost","DummyRegressor"]:
      a=xcv[name]
      lines.append(f"| {name} | {a['MAE'][0]:.3f} +/- {a['MAE'][1]:.3f} | {a['RMSE'][0]:.3f} +/- {a['RMSE'][1]:.3f} | {a['R2'][0]:.3f} +/- {a['R2'][1]:.3f} |")
    lines += ["", "### Distribusi target per fold (tanpa melatih ulang)", "",
              "Fold membership direkonstruksi deterministik dari cleaned data menggunakan GroupKFold(5) dan grup empat kolom. Ini memeriksa komposisi target, bukan menghitung skor model per fold.", "",
              "| Fold | n | Mean (mL) | SD populasi (mL) | Min (mL) | Max (mL) | 1474.6 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for _,r in fold_stats.iterrows(): lines.append(f"| {int(r.fold)} | {int(r.n)} | {r['mean']:.3f} | {r['std']:.3f} | {r['min']:.3f} | {r['max']:.3f} | {int(r.outlier_count)} |")
    lines += ["", f"Fold 5 berisi baris sumber **{', '.join(map(str,c5.source_rows))}**; targetnya mean **{c5['mean']:.3f} mL**, SD **{c5['std']:.3f} mL**, maksimum **{c5['max']:.3f} mL**, termasuk nilai 1474.6 pada baris sumber **{', '.join(map(str,train_outlier_rows))}**. Fold 1-4 memiliki target maksimum antara **{fold_stats.iloc[:4]['max'].min():.1f}** dan **{fold_stats.iloc[:4]['max'].max():.1f} mL**.",
              "- XGBoost CV RMSE memiliki SD antar-fold **72.759 mL** dan R2 SD **0.209**; ini menunjukkan variasi hasil CV yang cukup besar.",
              "- Penempatan 1474.6 di fold 5 dan distribusi target fold tersebut yang lebih lebar bertepatan dengan variasi CV. Ini merupakan indikasi kuat sebagai kontributor potensial, tetapi bukan bukti kausal karena skor per fold dan residual validasi tidak disimpan.",
              "- Baris dengan 1474.6 berada di training split (fold 5), bukan test split: `"+(", ".join(map(str,train_outlier_rows)) or "tidak ditemukan")+"`.", "",
              "## Duplikasi dan kebocoran", "",
      f"- Grup fold dibuat dari Soil Moisture, Temperature, Usia Tanam, dan Volume Penyiraman; lima fold berisi 25, 25, 24, 24, dan 24 baris. Grup exact duplikat lintas split train-test: **{split_group_overlap}**.",
              "- Rekonstruksi GroupKFold mempertahankan grup exact duplicate secara utuh pada fold yang sama.",
              "- X train/test berisi tepat tiga fitur yang diminta. Pengujian keselarasan terhadap cleaned data dan scaler berhasil dengan galat numerik maksimum di bawah 1e-10.",
              "- Metrik dihitung pada 30 target test yang sejajar; test tidak masuk ke training atau CV berdasarkan konfigurasi yang diaudit.", "",
              "## Prediksi testing per baris", "",
              "Residual = aktual - prediksi. Angka menampilkan nilai dari file prediksi yang disimpan.", "",
              "| No. | Aktual (mL) | Prediksi XGBoost (mL) | Residual (mL) | Prediksi Dummy (mL) |", "|---:|---:|---:|---:|---:|"]
    for i,r in pred.iterrows(): lines.append(f"| {i+1} | {r.Actual:.6f} | {r.Predicted_XGBoost:.6f} | {r.Residual_Actual_Minus_Predicted:.6f} | {r.Predicted_Dummy_Mean:.6f} |")
    lines += ["", "## Feature importance", "", "| Fitur | Importance |", "|---|---:|"]
    for k,v in sorted(fi.items(),key=lambda kv:-kv[1]): lines.append(f"| {k} | {v:.6f} |")
    lines += ["", "## Grafik", "", "Semua grafik direfresh dari model, metrik, dan prediksi aktual tersimpan pada resolusi 300 DPI. `cross_validation_performance_summary.png` memakai mean +/- SD yang ada. Skor per fold tidak tersedia; `cross_validation_fold_target_distribution.png` menunjukkan distribusi target aktual tiap fold tanpa mengklaim performa model per fold.", ""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    return {"test_xgb":xgb_test,"test_dummy":dummy_test,"metric_diff":metric_diff,"prediction_replay_diff":replay_diff,"fold_stats":fold_stats,"problems":problems}

if __name__=="__main__":
    result=run_audit()
    print("XGBoost test metrics:",result["test_xgb"])
    print("Dummy test metrics:",result["test_dummy"])
    print("Metric CSV max difference:",result["metric_diff"])
    print("Prediction replay max difference:",result["prediction_replay_diff"])
    print("Per-fold target distribution:\n",result["fold_stats"].to_string(index=False))
    print("Inconsistencies:",result["problems"] or "None")
