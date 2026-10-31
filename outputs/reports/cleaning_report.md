# Laporan Cleaning Data

**Sumber:** `data/processed/data_filtered.csv`.
**Metode:** konversi tipe numerik dan validasi; tidak ada baris dihapus dan tidak dilakukan imputasi.

## Hasil

- Baris sebelum cleaning: **152**.
- Baris setelah cleaning: **152**.
- Kolom fitur dikonversi menjadi numeric: `Soil Moisture`, `Temperature`, `Usia Tanam`, `Volume Penyiraman`. Temperature dengan koma desimal dikonversi ke float menggunakan titik desimal.
- `source_excel_row` dipertahankan sebagai integer; `potential_outlier` dan `duplicate_review` dipertahankan sebagai boolean.
- Kegagalan konversi nilai non-kosong: **0**.
  - `Soil Moisture`: 0.
  - `Temperature`: 0.
  - `Usia Tanam`: 0.
  - `Volume Penyiraman`: 0.

## Missing values

| Kolom | Sebelum | Sesudah |
|---|---:|---:|
| source_excel_row | 0 | 0 |
| Soil Moisture | 0 | 0 |
| Temperature | 0 | 0 |
| Usia Tanam | 0 | 0 |
| Volume Penyiraman | 0 | 0 |
| potential_outlier | 0 | 0 |
| duplicate_review | 0 | 0 |

## Validasi dan rekaman yang dipertahankan

- Semua nilai `Volume Penyiraman` non-negatif: **Ya**.
- Nilai potensial outlier 1474.6 dipertahankan pada baris sumber: **153**.
- Duplikat baris fitur identik tetap dipertahankan: **4 baris dalam 2 grup**.
  - Grup 1, baris sumber: 99, 101.
  - Grup 2, baris sumber: 134, 136.

## Output

- Data cleaned: `data/processed/data_cleaned.csv`.
- Workbook formatted: `data/processed/data_cleaned_formatted.xlsx` (worksheet `Cleaned Data`; scatter charts ditambahkan pada worksheet `Charts` setelah berhasil dibuat).
- Scatter plots: `outputs/figures/cleaned/` (300 DPI).
