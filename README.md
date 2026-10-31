# Prediksi Penyiraman Cabai

Project workspace for developing a chili pepper irrigation prediction system.

## Rasio training dan testing

Pipeline menggunakan rasio sasaran **70:30** (training:testing). Split dibuat dengan `GroupShuffleSplit(test_size=0.30, random_state=42)` dan mengelompokkan record duplikat identik agar tidak bocor ke kedua partisi. Karena pengelompokan tersebut, jumlah baris aktual dapat sedikit berbeda dari 70:30 tepat.

Jalankan tahapan berikut dari root project setelah mengubah proses split:

```powershell
python src/split_data.py
python src/normalize_data.py
python src/train_evaluate.py
python src/audit_model_evaluation.py
python src/export_comparison.py
python src/export_model_math.py
```

Model dilatih hanya pada training split; testing split digunakan untuk evaluasi akhir. Scaler di-fit pada fitur training saja.
