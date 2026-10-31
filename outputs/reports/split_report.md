# Laporan Train-Test Split

**Sumber:** `data/processed/data_cleaned.csv`.
**Metode:** `GroupShuffleSplit(test_size=0.30, random_state=42)`; grup dibentuk dari seluruh empat kolom dataset untuk mencegah duplikasi identik berada pada kedua split.

**Rasio sasaran:** 70% training dan 30% testing. Karena split menjaga grup duplikat tetap utuh, rasio aktual dihitung dari jumlah baris dan dapat sedikit berbeda dari rasio sasaran.

## Hasil

- Baris sumber: **152**.
- Training: **107** baris (70.39%).
- Testing: **45** baris (29.61%).
- Grup unik: **150**; grup training: **105**; grup testing: **45**.
- Fitur (X): `Soil Moisture`, `Temperature`, `Usia Tanam`.
- Target (y): `Volume Penyiraman`; tidak disertakan di X.
- Grup duplikat identik yang muncul lebih dari sekali: **4 baris dalam 2 grup**.
- Grup identik di kedua partisi: **0** (verifikasi lulus).

## Artefak

- `data/processed/splits/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.
- `outputs/figures/splits/training_testing_counts.png` dan `target_distribution_train_test.png` (300 DPI).
- Tiga scatter plot points-only untuk masing-masing partisi berada di `outputs/figures/train/` dan `outputs/figures/test/` (300 DPI).
- Data sumber cleaned tidak diubah. Tidak dilakukan normalisasi, training, atau evaluasi model.
