# Laporan Train-Test Split

**Sumber:** `data/processed/data_cleaned.csv`.
**Metode:** `GroupShuffleSplit(test_size=0.20, random_state=42)`; grup dibentuk dari seluruh empat kolom dataset.

## Hasil

- Baris sumber: **152**.
- Training: **122** baris (80.26%).
- Testing: **30** baris (19.74%).
- Grup unik: **150**; grup training: **120**; grup testing: **30**.
- Fitur (X): `Soil Moisture`, `Temperature`, `Usia Tanam`.
- Target (y): `Volume Penyiraman`; tidak disertakan di X.
- Grup duplikat identik yang muncul lebih dari sekali: **4 baris dalam 2 grup**.
- Grup identik di kedua partisi: **0** (verifikasi lulus).

## Artefak

- `data/processed/splits/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.
- `outputs/figures/splits/training_testing_counts.png` dan `target_distribution_train_test.png` (300 DPI).
- Tiga scatter plot points-only untuk masing-masing partisi berada di `outputs/figures/train/` dan `outputs/figures/test/` (300 DPI).
- Data sumber cleaned tidak diubah. Tidak dilakukan normalisasi, training, atau evaluasi model.
