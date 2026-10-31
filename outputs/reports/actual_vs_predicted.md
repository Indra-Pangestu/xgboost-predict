# Perbandingan Aktual dan Prediksi

Dataset memakai split sasaran training:testing **70:30**. Group split menghasilkan **107** baris training dan **45** testing dari 152 record.

Pada data testing, MAE = **35.111 mL**, RMSE = **59.388 mL**, dan R-squared = **0.917**.

Metrik dihitung dari prediksi XGBoost testing yang tersimpan dan disejajarkan dengan target `y_test`.

Grafik aktual dan prediksi per sampel tersedia pada `outputs/figures/model_evaluation/actual_vs_predicted_training_by_sample.png` dan `actual_vs_predicted_testing_by_sample.png`. Scatter aktual vs prediksi tersedia pada file `actual_vs_predicted_training.png` dan `actual_vs_predicted_testing.png`.

Tabel gabungan: `outputs/tables/complete_actual_vs_predicted.csv` dan `.xlsx`; workbook memuat ringkasan jumlah record. Tabel testing: `outputs/tables/actual_vs_predicted.csv` dan `.xlsx`.
