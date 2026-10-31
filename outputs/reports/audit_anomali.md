# Audit Anomali Dataset

**Sumber:** `data/raw/data_mentah.xlsx` - sheet `Sheet1`  
**Metode:** inspeksi baca-saja terhadap sel workbook dan baris tabel. Tidak ada data yang diubah, dibersihkan, atau dihapus.

## 1. Temperature

- Nilai unik mentah: **39**; tipe sel mentah terdiri dari angka dan teks.
- Daftar nilai unik mentah (urutan kemunculan):

```text
30.9, 34.2, '32,1', 31.9, 36.1, 32.5, 32.2, '28,5', '29,7', '30,5', 30.6, 31.6, 32.6, 33.6, '30,2', '31,3', '29,3', '31,4', 33.4, 27.2, 26.2, 27.3, 34.4, 35.3, 29.3, 30.3, 31.4, 26.3, 25.7, 28.4, 29.6, 27.4, 29.1, 29.4, 35.6, 36.3, 37.5, 37.9, 33.9
```
- Format tidak konsisten: angka desimal memakai titik sebagai numeric cell dan koma sebagai teks pada **36 baris**; nilai teks unik menggunakan koma desimal: '32,1', '28,5', '29,7', '30,5', '30,2', '31,3', '29,3', '31,4'.
- Dapat dikonversi secara aman secara sintaksis dengan mengganti koma desimal menjadi titik: **172/172** nilai berhasil diparse, tanpa nilai gagal. Ini tidak mengubah workbook; konversi mempertahankan nilai numerik yang dimaksud.

## 2. Volume Penyiraman

- Nilai negatif: **20**.
- Nilai nol: **0**.
- Nilai besar: ambang pencilan atas metode IQR (Q3 + 1.5 x IQR) = **948.975**; ditemukan **1** nilai di atas ambang.
- Seluruh nilai negatif (nomor baris adalah nomor baris Excel):

| Baris Excel | Soil Moisture | Temperature (mentah) | Usia Tanam | Volume Penyiraman |
|---:|---:|---:|---:|---:|
| 5 | 80.76163687014848 | 30.9 | 66 | -20.3999999999996 |
| 54 | 81.30257214724259 | 29,3 | 79 | -70.39999999999964 |
| 89 | 80.76163687014848 | 35.3 | 87 | -20.399999999999636 |
| 94 | 80.86982392556732 | 30.3 | 89 | -30.399999999999636 |
| 103 | 81.5189462580802 | 29.3 | 91 | -90.39999999999964 |
| 106 | 82.16806859059312 | 29.3 | 92 | -150.39999999999964 |
| 108 | 82.49262975684958 | 29.3 | 92 | -180.39999999999964 |
| 110 | 78.97655045573796 | 31.4 | 93 | -150.39999999999964 |
| 112 | 79.24701809428501 | 31.4 | 93 | -180.39999999999964 |
| 118 | 80.76163687014848 | 25.7 | 95 | -20.399999999999636 |
| 128 | 81.14029156411435 | 28.4 | 97 | -55.399999999999636 |
| 143 | 80.59935628702024 | 29.1 | 101 | -5.399999999999636 |
| 148 | 84.38590322667892 | 29.4 | 102 | -355.39999999999964 |
| 150 | 85.03502555919184 | 35.6 | 119 | -415.39999999999964 |
| 152 | 82.76309739539664 | 35.6 | 119 | -205.39999999999964 |
| 154 | 83.41221972790954 | 34.4 | 120 | -265.39999999999964 |
| 158 | 81.89760095204608 | 36.3 | 121 | -125.39999999999964 |
| 159 | 81.35666567495198 | 36.3 | 121 | -75.39999999999964 |
| 160 | 86.54964433505529 | 36.3 | 121 | -555.3999999999996 |
| 163 | 83.6826873664566 | 37.5 | 122 | -290.39999999999964 |

Nilai yang melewati ambang besar (baris Excel):

| Baris Excel | Volume Penyiraman |
|---:|---:|
| 153 | 1474.6000000000004 |

## 3. Duplikat

- Baris yang menjadi bagian dari duplikat: **4**; duplikat ekstra setelah satu baris per grup: **2**.
- Grup duplikat baris penuh yang identik (nomor baris Excel):

### Grup 1: baris 99, 101

| Soil Moisture | Temperature (mentah) | Usia Tanam | Volume Penyiraman |
|---:|---:|---:|---:|
| 78.70608281719092 | 31.6 | 90 | 169.60000000000036 |

### Grup 2: baris 134, 136

| Soil Moisture | Temperature (mentah) | Usia Tanam | Volume Penyiraman |
|---:|---:|---:|---:|
| 78.11105401238741 | 29.6 | 99 | 224.60000000000036 |

## 4. Jumlah dan baris kosong

- Baris non-kosong aktual: **172** (di luar header).
- Rentang baris data workbook: 2-173; baris kosong sepenuhnya dalam rentang tersebut: **0**.
- Pembacaan pandas menghasilkan **172 baris** dan tidak mengecualikan baris data non-kosong; tidak ada baris kosong yang perlu dibedakan dari record.
