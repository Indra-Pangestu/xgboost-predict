# Representasi Matematis Model XGBoost

## Sumber dan metode ekstraksi

Ekstraksi membaca `models/xgboost_model.joblib` dan `models/scaler.joblib`. Tidak ada pelatihan ulang maupun fit scaler. Semua struktur pohon, kondisi split, nilai daun, cover, dan statistik pohon berasal dari `Booster.get_dump()`.

## Ringkasan model

- Estimator: `XGBRegressor`; objektif: `reg:squarederror`.
- Jumlah pohon aktual: **250**.
- Base score konfigurasi booster: **322.0299**.
- Urutan fitur: `Soil Moisture`, `Temperature`, `Usia Tanam`.
- Versi XGBoost: `3.4.1`.

### Parameter estimator tersimpan

| Parameter | Nilai |
|---|---|
| `objective` | `reg:squarederror` |
| `base_score` | `None` |
| `booster` | `None` |
| `callbacks` | `None` |
| `colsample_bylevel` | `None` |
| `colsample_bynode` | `None` |
| `colsample_bytree` | `0.8` |
| `device` | `None` |
| `early_stopping_rounds` | `None` |
| `enable_categorical` | `True` |
| `eval_metric` | `None` |
| `feature_types` | `None` |
| `feature_weights` | `None` |
| `gamma` | `None` |
| `grow_policy` | `None` |
| `importance_type` | `None` |
| `interaction_constraints` | `None` |
| `learning_rate` | `0.03` |
| `max_bin` | `None` |
| `max_cat_threshold` | `None` |
| `max_cat_to_onehot` | `None` |
| `max_delta_step` | `None` |
| `max_depth` | `2` |
| `max_leaves` | `None` |
| `min_child_weight` | `3` |
| `missing` | `nan` |
| `monotone_constraints` | `None` |
| `multi_strategy` | `None` |
| `n_estimators` | `250` |
| `n_jobs` | `1` |
| `num_parallel_tree` | `None` |
| `random_state` | `42` |
| `reg_alpha` | `None` |
| `reg_lambda` | `5` |
| `sampling_method` | `None` |
| `scale_pos_weight` | `None` |
| `subsample` | `0.8` |
| `tree_method` | `None` |
| `validate_parameters` | `None` |
| `verbosity` | `0` |

## Persamaan StandardScaler

\[ z_j = \frac{x_j - \mu_j}{s_j} \]

`x_j` adalah nilai fitur pada skala asli, `μ_j` mean training, dan `s_j` simpangan baku populasi training (`StandardScaler.scale_`, ddof=0). Nilai mean dan skala berikut dibaca langsung dari scaler tersimpan:

| Fitur | Mean training (μ) | Std. deviasi training (s) |
|---|---:|---:|
| `Soil Moisture` | 77.063560653379042 | 2.1975308231676509 |
| `Temperature` | 30.998130841121494 | 2.8344997847256175 |
| `Usia Tanam` | 88.570093457943926 | 15.270221555249481 |

## Persamaan ensemble

\[ \hat{y} = b + \sum_{t=1}^{T} f_t(\mathbf{z}) \]

Di sini `b = 322.0299`, `T = 250`, dan `f_t(z)` adalah nilai daun yang dicapai pada pohon ke-`t`. Output model merupakan base score ditambah jumlah kontribusi seluruh pohon.

## Aturan IF–THEN

[`tree_rules.csv`](../tables/tree_rules.csv) memuat **872 aturan daun**. Setiap aturan menyatakan kondisi IF dari akar sampai daun dan kontribusi THEN. Kondisi menggunakan fitur yang telah ditransformasikan dengan StandardScaler; `≤` mengikuti cabang yes dan `>` mengikuti cabang no. Jalur nilai hilang mengikuti cabang missing yang tersimpan. Nilai THEN adalah kontribusi pohon, bukan prediksi akhir.

Dump seluruh pohon: [`xgboost_trees.json`](xgboost_trees.json) dan [`xgboost_trees.txt`](xgboost_trees.txt). Nilai ambang dan daun merupakan nilai yang diekspor XGBoost, bukan estimasi manual.

## Verifikasi

Pada **45** baris testing, model/scaler tersimpan dibandingkan dengan `outputs/tables/test_predictions.csv`; selisih absolut maksimum adalah **2.890625e-05 mL** (toleransi 3×10⁻⁵ mL untuk presisi serialisasi). Rekonstruksi `base score + leaf dari semua pohon` dibandingkan dengan prediksi model; selisih maksimum **0.000314905578 mL** (toleransi 5×10⁻⁴ mL karena digit presisi pada dump pohon). Kedua pemeriksaan lulus.
