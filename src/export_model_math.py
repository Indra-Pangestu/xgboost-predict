"""Extract the saved XGBoost model as mathematical/tree artifacts without fitting."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import DMatrix

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Soil Moisture", "Temperature", "Usia Tanam"]
TARGET = "Volume Penyiraman"


def serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def feature_name(key: str, names: list[str]) -> str:
    if key in names:
        return key
    if key.startswith("f") and key[1:].isdigit() and int(key[1:]) < len(names):
        return names[int(key[1:])]
    return key


def main() -> None:
    reports = ROOT / "outputs" / "reports"
    tables = ROOT / "outputs" / "tables"
    reports.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    model = joblib.load(ROOT / "models" / "xgboost_model.joblib")
    scaler = joblib.load(ROOT / "models" / "scaler.joblib")
    booster = model.get_booster()
    names = list(booster.feature_names or getattr(model, "feature_names_in_", FEATURES))
    if names != FEATURES:
        raise ValueError(f"Unexpected model feature names/order: {names}")

    config = json.loads(booster.save_config())
    base_raw = config["learner"]["learner_model_param"]["base_score"]
    base_score = float(str(base_raw).strip("[]"))
    params = {key: serializable(value) for key, value in model.get_params(deep=False).items()}
    json_dumps = booster.get_dump(dump_format="json", with_stats=True)
    text_dumps = booster.get_dump(dump_format="text", with_stats=True)
    if not json_dumps or len(json_dumps) != len(text_dumps):
        raise ValueError("Tree dump extraction failed")

    (reports / "xgboost_trees.json").write_text(
        json.dumps([json.loads(tree) for tree in json_dumps], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports / "xgboost_trees.txt").write_text(
        "\n\n".join(f"===== Tree {i} =====\n{tree}" for i, tree in enumerate(text_dumps)) + "\n",
        encoding="utf-8",
    )

    rules: list[dict] = []
    leaf_maps: list[dict[int, float]] = []

    def walk(node: dict, tree_id: int, path: list[str], leaf_map: dict[int, float]) -> None:
        node_id = int(node["nodeid"])
        if "leaf" in node:
            leaf_value = float(node["leaf"])
            leaf_map[node_id] = leaf_value
            rules.append({
                "Nomor Pohon": tree_id,
                "Nomor Daun": node_id,
                "Kondisi IF": " AND ".join(path) if path else "TRUE (tanpa split)",
                "Nilai THEN (kontribusi daun)": leaf_value,
                "Cover (jumlah Hessian)": float(node["cover"]) if "cover" in node else None,
            })
            return
        split_feature = feature_name(str(node["split"]), names)
        threshold = float(node["split_condition"])
        children = {int(child["nodeid"]): child for child in node["children"]}
        missing = int(node["missing"])
        for child_id, operator in ((int(node["yes"]), "<="), (int(node["no"]), ">")):
            condition = f"{split_feature} {operator} {threshold:.17g}"
            if child_id == missing:
                condition += " [termasuk nilai hilang]"
            walk(children[child_id], tree_id, path + [condition], leaf_map)

    for tree_id, dump in enumerate(json_dumps):
        leaf_map: dict[int, float] = {}
        walk(json.loads(dump), tree_id, [], leaf_map)
        leaf_maps.append(leaf_map)
    rules_path = tables / "tree_rules.csv"
    pd.DataFrame(rules).to_csv(rules_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    # Verify tree dump math independently: traverse the exported leaf IDs and sum the
    # exported leaf values with the configured base score, then compare with inference.
    split_dir = ROOT / "data" / "processed" / "splits"
    X_test = pd.read_csv(split_dir / "X_test.csv")
    if list(X_test.columns) != names:
        raise ValueError("Test feature names/order do not match model")
    X_scaled = scaler.transform(X_test[names])
    matrix = DMatrix(X_scaled, feature_names=names)
    leaf_indices = np.asarray(booster.predict(matrix, pred_leaf=True)).reshape(len(X_test), -1)
    extracted_formula = np.asarray([
        base_score + sum(leaf_maps[tree_id][int(leaf_indices[row, tree_id])]
                         for tree_id in range(len(leaf_maps)))
        for row in range(len(X_test))
    ])
    model_prediction = np.asarray(model.predict(pd.DataFrame(X_scaled, columns=names)))
    dump_replay_delta = float(np.max(np.abs(extracted_formula - model_prediction)))
    if dump_replay_delta > 5e-4:
        raise ValueError(f"Exported tree formula does not reproduce model prediction: {dump_replay_delta}")
    saved_test = pd.read_csv(tables / "test_predictions.csv")["Predicted_XGBoost"].to_numpy()
    saved_prediction_delta = float(np.max(np.abs(model_prediction - saved_test)))
    if len(saved_test) != len(model_prediction) or saved_prediction_delta > 3e-5:
        raise ValueError(f"Model predictions do not match saved test predictions: {saved_prediction_delta}")

    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    if len(means) != len(names) or len(scales) != len(names):
        raise ValueError("Scaler dimensions do not match model features")
    param_lines = [f"| `{key}` | `{value}` |" for key, value in params.items()]
    scaler_lines = [f"| `{name}` | {means[i]:.17g} | {scales[i]:.17g} |" for i, name in enumerate(names)]
    report = [
        "# Representasi Matematis Model XGBoost", "",
        "## Sumber dan metode ekstraksi", "",
        "Ekstraksi membaca `models/xgboost_model.joblib` dan `models/scaler.joblib`. Tidak ada pelatihan ulang maupun fit scaler. Semua struktur pohon, kondisi split, nilai daun, cover, dan statistik pohon berasal dari `Booster.get_dump()`.", "",
        "## Ringkasan model", "",
        f"- Estimator: `{type(model).__name__}`; objektif: `{model.get_xgb_params().get('objective')}`.",
        f"- Jumlah pohon aktual: **{len(json_dumps)}**.",
        f"- Base score konfigurasi booster: **{base_score:.17g}**.",
        f"- Urutan fitur: {', '.join(f'`{name}`' for name in names)}.",
        f"- Versi XGBoost: `{xgboost.__version__}`.", "",
        "### Parameter estimator tersimpan", "",
        "| Parameter | Nilai |", "|---|---|", *param_lines, "",
        "## Persamaan StandardScaler", "",
        r"\[ z_j = \frac{x_j - \mu_j}{s_j} \]", "",
        "`x_j` adalah nilai fitur pada skala asli, `μ_j` mean training, dan `s_j` simpangan baku populasi training (`StandardScaler.scale_`, ddof=0). Nilai mean dan skala berikut dibaca langsung dari scaler tersimpan:", "",
        "| Fitur | Mean training (μ) | Std. deviasi training (s) |", "|---|---:|---:|", *scaler_lines, "",
        "## Persamaan ensemble", "",
        r"\[ \hat{y} = b + \sum_{t=1}^{T} f_t(\mathbf{z}) \]", "",
        f"Di sini `b = {base_score:.17g}`, `T = {len(json_dumps)}`, dan `f_t(z)` adalah nilai daun yang dicapai pada pohon ke-`t`. Output model merupakan base score ditambah jumlah kontribusi seluruh pohon.", "",
        "## Aturan IF–THEN", "",
        f"[`tree_rules.csv`](../tables/tree_rules.csv) memuat **{len(rules)} aturan daun**. Setiap aturan menyatakan kondisi IF dari akar sampai daun dan kontribusi THEN. Kondisi menggunakan fitur yang telah ditransformasikan dengan StandardScaler; `≤` mengikuti cabang yes dan `>` mengikuti cabang no. Jalur nilai hilang mengikuti cabang missing yang tersimpan. Nilai THEN adalah kontribusi pohon, bukan prediksi akhir.", "",
        "Dump seluruh pohon: [`xgboost_trees.json`](xgboost_trees.json) dan [`xgboost_trees.txt`](xgboost_trees.txt). Nilai ambang dan daun merupakan nilai yang diekspor XGBoost, bukan estimasi manual.", "",
        "## Verifikasi", "",
        f"Pada **{len(X_test)}** baris testing, model/scaler tersimpan dibandingkan dengan `outputs/tables/test_predictions.csv`; selisih absolut maksimum adalah **{saved_prediction_delta:.9g} mL** (toleransi 3×10⁻⁵ mL untuk presisi serialisasi). Rekonstruksi `base score + leaf dari semua pohon` dibandingkan dengan prediksi model; selisih maksimum **{dump_replay_delta:.9g} mL** (toleransi 5×10⁻⁴ mL karena digit presisi pada dump pohon). Kedua pemeriksaan lulus.", "",
    ]
    (reports / "model_matematika.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Trees={len(json_dumps)}; base_score={base_score:.10g}; leaf rules={len(rules)}")
    print(f"Saved-prediction delta={saved_prediction_delta:.9g}; extracted-formula delta={dump_replay_delta:.9g}")


if __name__ == "__main__":
    main()
