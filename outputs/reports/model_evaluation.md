# Audit Hasil Evaluasi XGBoost

## Status artefak

- Model tersimpan dan dapat dimuat sebagai `XGBRegressor`; versi library saat audit: XGBoost 3.4.1.
- File prediksi dan metrik tersedia. Split dan scaler tersedia; audit ini tidak menulis ulang data tersebut.
- Baris prediksi: **45**; actual pada seluruh baris cocok berurutan dengan `y_test`: **Ya**.
- Prediksi ulang memakai model tersimpan berbeda maksimum **2.890625e-05** dari nilai CSV; perbedaan kecil konsisten dengan pembulatan/serialisasi float.
- Kolom residual tersimpan berbeda maksimum **2.890625e-05** dari actual dikurangi prediksi CSV.
- Fitur model: `Soil Moisture`, `Temperature`, `Usia Tanam`; target tidak terdapat di input X.
- Row alignment terhadap cleaned data dan scaler: max error X train **4.44e-16**, X test **4.44e-16**, y train **0**, y test **0**.
- Prediksi negatif tersimpan: **0**; nilai disajikan tanpa clipping.

## Parameter model tersimpan

`n_estimators=250`, `max_depth=2`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`, `reg_lambda=5`, `objective=reg:squarederror`, `random_state=42`.

## Verifikasi metrik testing

Metrik dihitung ulang dari 45 baris pada `test_predictions.csv`; selisih terhadap `model_metrics.csv` dicatat di bawah. Semua nilai dalam satuan asli mL kecuali R2. Split sasaran adalah 70:30.

| Metrik | XGB tersimpan | XGB hitung ulang | Selisih | Dummy tersimpan | Dummy hitung ulang | Selisih |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 35.1110719638 | 35.1110730444 | 1.08e-06 | 174.6282450675 | 174.6282450675 | 0 |
| RMSE | 59.3881599054 | 59.3881607433 | 8.38e-07 | 209.6844808198 | 209.6844808198 | 0 |
| R2 | 0.9165556045 | 0.9165556021 | 2.35e-09 | -0.0402315261 | -0.0402315261 | 6.25e-17 |

Kinerja XGBoost lebih baik daripada baseline mean pada split testing: MAE turun **79.9%** dan RMSE turun **71.7%**. Nilai R2 testing XGBoost adalah **0.917**.

## Cross-validation GroupKFold

Metrik CV yang tersimpan hanya berisi mean dan standar deviasi antar lima fold; skor MAE/RMSE/R2 per fold tidak tersimpan. Karena audit ini tidak melatih ulang fold model, skor per fold tidak dapat direkonstruksi secara sah. Grafik yang tersedia menampilkan ringkasan mean +/- SD, bukan skor per fold.

| Model | MAE mean +/- SD | RMSE mean +/- SD | R2 mean +/- SD |
|---|---:|---:|---:|
| XGBoost | 35.225 +/- 16.568 | 77.136 +/- 70.028 | 0.836 +/- 0.179 |
| DummyRegressor | 151.531 +/- 25.468 | 200.733 +/- 57.371 | -0.129 +/- 0.202 |

### Distribusi target per fold (tanpa melatih ulang)

Fold membership direkonstruksi deterministik dari cleaned data menggunakan GroupKFold(5) dan grup empat kolom. Ini memeriksa komposisi target, bukan menghitung skor model per fold.

| Fold | n | Mean (mL) | SD populasi (mL) | Min (mL) | Max (mL) | 1474.6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 240.055 | 148.162 | 4.600 | 534.600 | 0 |
| 2 | 22 | 354.827 | 152.766 | 59.600 | 609.600 | 0 |
| 3 | 21 | 317.933 | 184.960 | 14.600 | 764.600 | 0 |
| 4 | 21 | 311.267 | 177.619 | 19.600 | 599.600 | 0 |
| 5 | 21 | 388.410 | 290.184 | 79.600 | 1474.600 | 1 |

Fold 5 berisi baris sumber **4, 11, 24, 36, 42, 49, 56, 63, 70, 78, 85, 95, 102, 109, 117, 125, 131, 139, 153, 157, 171**; targetnya mean **388.410 mL**, SD **290.184 mL**, maksimum **1474.600 mL**.
- Ringkasan mean dan SD CV yang dilaporkan berasal dari artefak evaluasi model yang tersimpan; nilai tersebut tidak menyimpan skor individual per fold.
- Baris dengan 1474.6 berada di training split (fold 5), bukan test split: `153`.

## Duplikasi dan kebocoran

- Grup fold dibuat dari Soil Moisture, Temperature, Usia Tanam, dan Volume Penyiraman; ukuran fold aktual: 22, 22, 21, 21, 21 baris. Grup exact duplikat lintas split train-test: **0**.
- Rekonstruksi GroupKFold mempertahankan grup exact duplicate secara utuh pada fold yang sama.
- X train/test berisi tepat tiga fitur yang diminta. Pengujian keselarasan terhadap cleaned data dan scaler berhasil dengan galat numerik maksimum di bawah 1e-10.
- Metrik dihitung pada 45 target test yang sejajar; test tidak masuk ke training atau CV berdasarkan konfigurasi yang diaudit.

## Prediksi testing per baris

Residual = aktual - prediksi. Angka menampilkan nilai dari file prediksi yang disimpan.

| No. | Aktual (mL) | Prediksi XGBoost (mL) | Residual (mL) | Prediksi Dummy (mL) |
|---:|---:|---:|---:|---:|
| 1 | 469.600000 | 453.634060 | 15.965936 | 322.029907 |
| 2 | 439.600000 | 448.171050 | -8.571051 | 322.029907 |
| 3 | 459.600000 | 453.889950 | 5.710046 | 322.029907 |
| 4 | 354.600000 | 341.581450 | 13.018549 | 322.029907 |
| 5 | 424.600000 | 429.572900 | -4.972906 | 322.029907 |
| 6 | 289.600000 | 393.245330 | -103.645331 | 322.029907 |
| 7 | 19.600000 | 154.239120 | -134.639120 | 322.029907 |
| 8 | 54.600000 | 159.900070 | -105.300070 | 322.029907 |
| 9 | 39.600000 | 49.167027 | -9.567027 | 322.029907 |
| 10 | 34.600000 | 49.167027 | -14.567027 | 322.029907 |
| 11 | 654.600000 | 802.162900 | -147.562903 | 322.029907 |
| 12 | 499.600000 | 498.688100 | 0.911890 | 322.029907 |
| 13 | 554.600000 | 527.761350 | 26.838647 | 322.029907 |
| 14 | 769.600000 | 799.603400 | -30.003394 | 322.029907 |
| 15 | 689.600000 | 799.196040 | -109.596045 | 322.029907 |
| 16 | 499.600000 | 498.280760 | 1.319238 | 322.029907 |
| 17 | 209.600000 | 205.463030 | 4.136972 | 322.029907 |
| 18 | 19.600000 | 44.278070 | -24.678069 | 322.029907 |
| 19 | 429.600000 | 439.154330 | -9.554327 | 322.029907 |
| 20 | 494.600000 | 489.084350 | 5.515649 | 322.029907 |
| 21 | 319.600000 | 323.186900 | -3.586890 | 322.029907 |
| 22 | 664.600000 | 806.696500 | -142.096472 | 322.029907 |
| 23 | 554.600000 | 537.892940 | 16.707056 | 322.029907 |
| 24 | 619.600000 | 809.735050 | -190.135046 | 322.029907 |
| 25 | 379.600000 | 396.160580 | -16.560583 | 322.029907 |
| 26 | 144.600000 | 189.611330 | -45.011328 | 322.029907 |
| 27 | 204.600000 | 207.456900 | -2.856894 | 322.029907 |
| 28 | 674.600000 | 804.765600 | -130.165625 | 322.029907 |
| 29 | 284.600000 | 280.448400 | 4.151605 | 322.029907 |
| 30 | 324.600000 | 313.591160 | 11.008844 | 322.029907 |
| 31 | 169.600000 | 184.445400 | -14.845404 | 322.029907 |
| 32 | 519.600000 | 508.162500 | 11.437494 | 322.029907 |
| 33 | 334.600000 | 327.043980 | 7.556024 | 322.029907 |
| 34 | 29.600000 | 49.292763 | -19.692763 | 322.029907 |
| 35 | 344.600000 | 329.566440 | 15.033563 | 322.029907 |
| 36 | 319.600000 | 303.939330 | 15.660669 | 322.029907 |
| 37 | 414.600000 | 396.187350 | 18.412653 | 322.029907 |
| 38 | 334.600000 | 315.699520 | 18.900476 | 322.029907 |
| 39 | 229.600000 | 241.208950 | -11.608954 | 322.029907 |
| 40 | 579.600000 | 526.932100 | 52.667871 | 322.029907 |
| 41 | 54.600000 | 56.725840 | -2.125842 | 322.029907 |
| 42 | 404.600000 | 418.152620 | -13.552618 | 322.029907 |
| 43 | 474.600000 | 494.748050 | -20.148047 | 322.029907 |
| 44 | 509.600000 | 504.481700 | 5.118311 | 322.029907 |
| 45 | 49.600000 | 64.483010 | -14.883009 | 322.029907 |

## Feature importance

| Fitur | Importance |
|---|---:|
| Soil Moisture | 0.734456 |
| Temperature | 0.144574 |
| Usia Tanam | 0.120970 |

## Grafik

Semua grafik direfresh dari model, metrik, dan prediksi aktual tersimpan pada resolusi 300 DPI. `cross_validation_performance_summary.png` memakai mean +/- SD yang ada. Skor per fold tidak tersedia; `cross_validation_fold_target_distribution.png` menunjukkan distribusi target aktual tiap fold tanpa mengklaim performa model per fold.
