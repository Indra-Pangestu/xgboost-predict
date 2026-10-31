"""Predict irrigation volume from three CLI feature values."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
TRAIN_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "X_train.csv"
FEATURES = ["Soil Moisture", "Temperature", "Usia Tanam"]


def parse_numeric(value: str) -> float:
    """Parse a finite numeric value; accept a comma as decimal separator."""
    cleaned = value.strip()
    if "," in cleaned and "." in cleaned:
        raise argparse.ArgumentTypeError("gunakan satu pemisah desimal saja (titik atau koma)")
    try:
        number = float(cleaned.replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("nilai harus berupa angka") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("nilai harus berupa angka terbatas, bukan NaN/inf")
    return number


def load_artifacts():
    for path in (MODEL_PATH, SCALER_PATH, TRAIN_FEATURES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Berkas yang dibutuhkan tidak ditemukan: {path}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    training = pd.read_csv(TRAIN_FEATURES_PATH)
    if list(training.columns) != FEATURES:
        raise ValueError(f"Urutan kolom data training tidak sesuai: {list(training.columns)}")
    model_features = list(getattr(model, "feature_names_in_", FEATURES))
    scaler_features = list(getattr(scaler, "feature_names_in_", FEATURES))
    if model_features != FEATURES:
        raise ValueError(f"Urutan fitur model tidak sesuai: {model_features}")
    if scaler_features != FEATURES:
        raise ValueError(f"Urutan fitur scaler tidak sesuai: {scaler_features}")
    ranges = {feature: (float(training[feature].min()), float(training[feature].max()))
              for feature in FEATURES}
    return model, scaler, ranges


def predict_irrigation(soil_moisture: float, temperature: float, plant_age: float):
    """Scale one input row and return its raw model prediction and range warnings."""
    values = [soil_moisture, temperature, plant_age]
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError("Semua input harus berupa angka terbatas")
    model, scaler, ranges = load_artifacts()
    row = pd.DataFrame([values], columns=FEATURES)
    scaled = scaler.transform(row[FEATURES])
    scaled_row = pd.DataFrame(scaled, columns=FEATURES)
    prediction = float(model.predict(scaled_row)[0])
    out_of_range = []
    for feature, value in zip(FEATURES, values):
        lower, upper = ranges[feature]
        if value < lower or value > upper:
            out_of_range.append((feature, value, lower, upper))
    return prediction, out_of_range


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prediksi volume penyiraman cabai dari tiga parameter input."
    )
    parser.add_argument("--soil-moisture", required=True, type=parse_numeric,
                        help="Kelembapan tanah")
    parser.add_argument("--temperature", required=True, type=parse_numeric,
                        help="Suhu; dapat memakai koma atau titik desimal")
    parser.add_argument("--plant-age", required=True, type=parse_numeric,
                        help="Usia tanam")
    args = parser.parse_args(argv)
    try:
        prediction, warnings = predict_irrigation(
            args.soil_moisture, args.temperature, args.plant_age
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    for feature, value, lower, upper in warnings:
        print(f"Peringatan: {feature}={value:g} berada di luar rentang training [{lower:g}, {upper:g}].",
              file=sys.stderr)
    if prediction < 0:
        print("Peringatan: prediksi mentah bernilai negatif; nilai tidak dipotong atau diubah.",
              file=sys.stderr)
    print(f"Prediksi volume penyiraman: {prediction:.2f} mL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
