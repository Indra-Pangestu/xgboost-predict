# Perbandingan Aktual dan Prediksi

Dari **30 sampel testing**, MAE sebesar **30.53 mL**, RMSE **49.57 mL**, dan R-squared **0.941**. Nilai R-squared menunjukkan prediksi mengikuti variasi volume aktual dengan baik pada sampel testing ini. Rata-rata selisih absolut prediksi dari aktual adalah sekitar 30.53 mL; RMSE yang lebih besar dari MAE mencerminkan pengaruh beberapa kesalahan yang lebih besar.

Metrik dihitung dari nilai yang tersimpan di `outputs/tables/test_predictions.csv`; seluruh **30** record disertakan. Tidak ada model yang dilatih ulang dan dataset tidak diubah.

- **MAE:** 30.532088 mL
- **RMSE:** 49.570132 mL
- **R-squared:** 0.940534

Berkas perbandingan: `outputs/tables/actual_vs_predicted.xlsx` dan `outputs/tables/actual_vs_predicted.csv`. Grafik 300 DPI tersimpan di `outputs/figures/model_evaluation/actual_vs_predicted_line.png` dan `actual_vs_predicted_scatter.png`.
