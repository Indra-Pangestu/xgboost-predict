# Laporan Normalisasi

**Metode:** `StandardScaler`; scaler di-fit hanya pada `X_train` lalu digunakan untuk mentransformasi `X_test`.
**Fitur:** Soil Moisture, Temperature, Usia Tanam. Target Volume Penyiraman tidak dinormalisasi.

## Ukuran split

- Training: **122** baris.
- Testing: **30** baris.

## Verifikasi statistik training

Rata-rata dan standar deviasi populasi (ddof=0) setelah normalisasi:

| Fitur | Mean | Std deviasi |
|---|---:|---:|
| Soil Moisture | 0.000000000000 | 1.000000000000 |
| Temperature | -0.000000000000 | 1.000000000000 |
| Usia Tanam | -0.000000000000 | 1.000000000000 |

Mean mendekati 0 dan standar deviasi mendekati 1 untuk semua fitur training.

## Artefak

- `data/processed/splits/X_train_normalized.csv` dan `X_test_normalized.csv`.
- `models/scaler.joblib`.
- `data/processed/normalized_data_formatted.xlsx` dengan worksheet data X/y dan `Charts`.
- `outputs/figures/normalized/feature_distributions_before.png` dan `feature_distributions_after.png`.
- Scatter points-only training/testing: `outputs/figures/normalized/` (masing-masing tiga grafik, 300 DPI).
- `y_train.csv`, `y_test.csv`, split asli, dan cleaned dataset tidak diubah.
- Tidak dilakukan pelatihan model XGBoost.
