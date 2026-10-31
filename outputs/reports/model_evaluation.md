# Audit Hasil Evaluasi XGBoost

## Status artefak

- Model tersimpan dan dapat dimuat sebagai `XGBRegressor`; versi library saat audit: XGBoost 3.4.1.
- File prediksi dan metrik tersedia. Split dan scaler tersedia; audit ini tidak menulis ulang data tersebut.
- Baris prediksi: **30**; actual pada seluruh baris cocok berurutan dengan `y_test`: **Ya**.
- Prediksi ulang memakai model tersimpan berbeda maksimum **2.5390625e-05** dari nilai CSV; perbedaan kecil konsisten dengan pembulatan/serialisasi float.
- Kolom residual tersimpan berbeda maksimum **2.5390625e-05** dari actual dikurangi prediksi CSV.
- Fitur model: `Soil Moisture`, `Temperature`, `Usia Tanam`; target tidak terdapat di input X.
- Row alignment terhadap cleaned data dan scaler: max error X train **2.22e-16**, X test **2.22e-16**, y train **0**, y test **0**.
- Prediksi negatif tersimpan: **0**; nilai disajikan tanpa clipping.

## Parameter model tersimpan

`n_estimators=250`, `max_depth=2`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`, `reg_lambda=5`, `objective=reg:squarederror`, `random_state=42`.

## Verifikasi metrik testing

Metrik dihitung ulang dari 30 baris pada `test_predictions.csv`; selisih terhadap `model_metrics.csv` dicatat di bawah. Semua nilai dalam satuan asli mL kecuali R2.

| Metrik | XGB tersimpan | XGB hitung ulang | Selisih | Dummy tersimpan | Dummy hitung ulang | Selisih |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 30.5320878601 | 30.5320878000 | 6.01e-08 | 169.3579234973 | 169.3579234973 | 0 |
| RMSE | 49.5701316894 | 49.5701322966 | 6.07e-07 | 211.6088604434 | 211.6088604434 | 2.84e-14 |
| R2 | 0.9405336386 | 0.9405336372 | 1.46e-09 | -0.0836746577 | -0.0836746577 | 4.16e-17 |

Kinerja XGBoost lebih baik daripada baseline mean pada split testing: MAE turun **82.0%** dan RMSE turun **76.6%**. Nilai R2 testing XGBoost adalah **0.941**.

## Cross-validation GroupKFold

Metrik CV yang tersimpan hanya berisi mean dan standar deviasi antar lima fold; skor MAE/RMSE/R2 per fold tidak tersimpan. Karena audit ini tidak melatih ulang fold model, skor tiap fold (termasuk fold 5) tidak dapat direkonstruksi secara sah. Grafik yang tersedia menampilkan ringkasan mean +/- SD, bukan skor per fold.

| Model | MAE mean +/- SD | RMSE mean +/- SD | R2 mean +/- SD |
|---|---:|---:|---:|
| XGBoost | 33.793 +/- 21.351 | 71.666 +/- 72.759 | 0.850 +/- 0.209 |
| DummyRegressor | 155.967 +/- 24.802 | 200.588 +/- 59.478 | -0.075 +/- 0.084 |

### Distribusi target per fold (tanpa melatih ulang)

Fold membership direkonstruksi deterministik dari cleaned data menggunakan GroupKFold(5) dan grup empat kolom. Ini memeriksa komposisi target, bukan menghitung skor model per fold.

| Fold | n | Mean (mL) | SD populasi (mL) | Min (mL) | Max (mL) | 1474.6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 288.800 | 176.469 | 4.600 | 574.600 | 0 |
| 2 | 25 | 287.400 | 170.957 | 14.600 | 649.600 | 0 |
| 3 | 24 | 300.642 | 170.006 | 19.600 | 609.600 | 0 |
| 4 | 24 | 312.100 | 165.592 | 19.600 | 599.600 | 0 |
| 5 | 24 | 427.100 | 277.541 | 24.600 | 1474.600 | 1 |

Fold 5 berisi baris sumber **4, 10, 17, 24, 30, 38, 44, 50, 56, 63, 69, 76, 84, 91, 97, 104, 113, 120, 130, 135, 141, 153, 168, 172**; targetnya mean **427.100 mL**, SD **277.541 mL**, maksimum **1474.600 mL**, termasuk nilai 1474.6 pada baris sumber **153**. Fold 1-4 memiliki target maksimum antara **574.6** dan **649.6 mL**.
- XGBoost CV RMSE memiliki SD antar-fold **72.759 mL** dan R2 SD **0.209**; ini menunjukkan variasi hasil CV yang cukup besar.
- Penempatan 1474.6 di fold 5 dan distribusi target fold tersebut yang lebih lebar bertepatan dengan variasi CV. Ini merupakan indikasi kuat sebagai kontributor potensial, tetapi bukan bukti kausal karena skor per fold dan residual validasi tidak disimpan.
- Baris dengan 1474.6 berada di training split (fold 5), bukan test split: `153`.

## Duplikasi dan kebocoran

- Grup fold dibuat dari Soil Moisture, Temperature, Usia Tanam, dan Volume Penyiraman; lima fold berisi 25, 25, 24, 24, dan 24 baris. Grup exact duplikat lintas split train-test: **0**.
- Rekonstruksi GroupKFold mempertahankan grup exact duplicate secara utuh pada fold yang sama.
- X train/test berisi tepat tiga fitur yang diminta. Pengujian keselarasan terhadap cleaned data dan scaler berhasil dengan galat numerik maksimum di bawah 1e-10.
- Metrik dihitung pada 30 target test yang sejajar; test tidak masuk ke training atau CV berdasarkan konfigurasi yang diaudit.

## Prediksi testing per baris

Residual = aktual - prediksi. Angka menampilkan nilai dari file prediksi yang disimpan.

| No. | Aktual (mL) | Prediksi XGBoost (mL) | Residual (mL) | Prediksi Dummy (mL) |
|---:|---:|---:|---:|---:|
| 1 | 439.600000 | 451.380650 | -11.780646 | 322.632787 |
| 2 | 424.600000 | 432.504000 | -7.903998 | 322.632787 |
| 3 | 54.600000 | 80.928650 | -26.328650 | 322.632787 |
| 4 | 39.600000 | 36.432934 | 3.167066 | 322.632787 |
| 5 | 654.600000 | 795.639650 | -141.039648 | 322.632787 |
| 6 | 554.600000 | 535.331240 | 19.268762 | 322.632787 |
| 7 | 769.600000 | 796.676000 | -27.076025 | 322.632787 |
| 8 | 689.600000 | 795.761540 | -106.161536 | 322.632787 |
| 9 | 209.600000 | 208.486530 | 1.113474 | 322.632787 |
| 10 | 429.600000 | 442.133450 | -12.533453 | 322.632787 |
| 11 | 319.600000 | 330.647060 | -11.047064 | 322.632787 |
| 12 | 664.600000 | 801.807800 | -137.207800 | 322.632787 |
| 13 | 554.600000 | 535.333560 | 19.266443 | 322.632787 |
| 14 | 379.600000 | 398.823730 | -19.223730 | 322.632787 |
| 15 | 144.600000 | 188.732100 | -44.132101 | 322.632787 |
| 16 | 204.600000 | 210.697780 | -6.097784 | 322.632787 |
| 17 | 674.600000 | 797.774540 | -123.174536 | 322.632787 |
| 18 | 284.600000 | 266.890530 | 17.709467 | 322.632787 |
| 19 | 324.600000 | 311.887420 | 12.712579 | 322.632787 |
| 20 | 334.600000 | 322.564300 | 12.035699 | 322.632787 |
| 21 | 29.600000 | 42.688084 | -13.088084 | 322.632787 |
| 22 | 344.600000 | 324.913360 | 19.686639 | 322.632787 |
| 23 | 319.600000 | 304.201540 | 15.398462 | 322.632787 |
| 24 | 414.600000 | 398.030880 | 16.569116 | 322.632787 |
| 25 | 334.600000 | 317.363160 | 17.236841 | 322.632787 |
| 26 | 229.600000 | 247.586230 | -17.986227 | 322.632787 |
| 27 | 579.600000 | 552.167400 | 27.432581 | 322.632787 |
| 28 | 54.600000 | 47.825836 | 6.774164 | 322.632787 |
| 29 | 474.600000 | 495.249420 | -20.649420 | 322.632787 |
| 30 | 509.600000 | 507.439360 | 2.160638 | 322.632787 |

## Feature importance

| Fitur | Importance |
|---|---:|
| Soil Moisture | 0.773743 |
| Temperature | 0.137071 |
| Usia Tanam | 0.089185 |

## Grafik

Semua grafik direfresh dari model, metrik, dan prediksi aktual tersimpan pada resolusi 300 DPI. `cross_validation_performance_summary.png` memakai mean +/- SD yang ada. Skor per fold tidak tersedia; `cross_validation_fold_target_distribution.png` menunjukkan distribusi target aktual tiap fold tanpa mengklaim performa model per fold.
