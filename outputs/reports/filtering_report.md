# Laporan Filtering Data

**Sumber:** `data/raw/data_mentah.xlsx` (sheet pertama).
**Aturan:** mengecualikan baris dengan `Volume Penyiraman < 0`; nilai lainnya dipertahankan.
**Perubahan sumber:** workbook asli tidak diubah.

## Hasil

- Baris sebelum filtering: **172**.
- Baris dikeluarkan: **20**.
- Baris tersisa: **152** (sesuai jumlah yang diharapkan: 152).
- Nilai 1474.6 dipertahankan pada baris Excel **153** dan ditandai `potential_outlier=true` (ambang IQR audit: 948.975).
- Duplikat persis dipertahankan dan ditandai `duplicate_review=true`: baris Excel **99, 101, 134, 136**.
- Tidak dilakukan pembersihan, normalisasi, split data, atau pelatihan model.

## Baris yang dikeluarkan

Nomor berikut adalah nomor baris Excel termasuk header; semua dikeluarkan karena `Volume Penyiraman` negatif:

| Baris Excel | Volume Penyiraman | Alasan |
|---:|---:|---|
| 5 | -20.3999999999996 | Negative Volume Penyiraman |
| 54 | -70.39999999999964 | Negative Volume Penyiraman |
| 89 | -20.399999999999636 | Negative Volume Penyiraman |
| 94 | -30.399999999999636 | Negative Volume Penyiraman |
| 103 | -90.39999999999964 | Negative Volume Penyiraman |
| 106 | -150.39999999999964 | Negative Volume Penyiraman |
| 108 | -180.39999999999964 | Negative Volume Penyiraman |
| 110 | -150.39999999999964 | Negative Volume Penyiraman |
| 112 | -180.39999999999964 | Negative Volume Penyiraman |
| 118 | -20.399999999999636 | Negative Volume Penyiraman |
| 128 | -55.399999999999636 | Negative Volume Penyiraman |
| 143 | -5.399999999999636 | Negative Volume Penyiraman |
| 148 | -355.39999999999964 | Negative Volume Penyiraman |
| 150 | -415.39999999999964 | Negative Volume Penyiraman |
| 152 | -205.39999999999964 | Negative Volume Penyiraman |
| 154 | -265.39999999999964 | Negative Volume Penyiraman |
| 158 | -125.39999999999964 | Negative Volume Penyiraman |
| 159 | -75.39999999999964 | Negative Volume Penyiraman |
| 160 | -555.3999999999996 | Negative Volume Penyiraman |
| 163 | -290.39999999999964 | Negative Volume Penyiraman |

## Berkas keluaran

- Data tersaring: `data/processed/data_filtered.csv` (menyertakan `source_excel_row`, `potential_outlier`, dan `duplicate_review`).
- Baris dikeluarkan: `outputs/tables/filtered_out_rows.csv` (menyertakan alasan pengeluaran).
- Grafik jumlah baris: `outputs/figures/row_counts_before_after.png` (300 DPI).
- Grafik distribusi target: `outputs/figures/target_distribution_before_after.png` (300 DPI).
