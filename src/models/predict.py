import json
import numpy as np
import pandas as pd
import joblib
import os

from src.models.train import load_model

# German Credit: 0 = good (bajo riesgo), 1 = bad (alto riesgo)
PROCESSED_DIR = "data/processed"


def _load_preprocessors():
    scaler  = joblib.load(os.path.join(PROCESSED_DIR, "scaler.pkl"))
    ord_enc = joblib.load(os.path.join(PROCESSED_DIR, "ordinal_encoder.pkl"))
    with open(os.path.join(PROCESSED_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    return scaler, ord_enc, feature_columns


def _preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Aplica el mismo pipeline del notebook 02 sobre un input crudo.
    input_data debe tener las columnas originales (sin encodear):
      Age, Credit amount, Duration, Job,
      Saving accounts, Checking account,
      Sex, Housing, Purpose
    """
    scaler, ord_enc, feature_columns = _load_preprocessors()

    df = pd.DataFrame([input_data])

    # --- Ordinal encoding ---
    ordinal_cols = ["Saving accounts", "Checking account"]
    df[ordinal_cols] = ord_enc.transform(df[ordinal_cols])

    # --- One-Hot Encoding ---
    nominal_cols = ["Sex", "Housing", "Purpose"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    # --- Alinear columnas con las del train ---
    df = df.reindex(columns=feature_columns, fill_value=0)

    # --- Escalar numéricas ---
    num_cols = ["Age", "Credit amount", "Duration"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def predict(input_data: dict, model_name: str = "best_model") -> dict:
    """
    Predicción de riesgo crediticio a partir de datos crudos (sin encodear).

    Args:
        input_data: dict con los campos originales del formulario.
                    Ejemplo:
                    {
                        "Age": 35,
                        "Sex": "male",
                        "Job": 2,
                        "Housing": "own",
                        "Saving accounts": "little",
                        "Checking account": "moderate",
                        "Credit amount": 5000,
                        "Duration": 24,
                        "Purpose": "car"
                    }
        model_name: nombre del checkpoint en models/checkpoints/.

    Returns:
        dict con risk_label, probability_default y probabilities.
    """
    model = load_model(model_name)
    X = _preprocess_input(input_data)

    proba = model.predict_proba(X)[0]
    prob_default = float(proba[1])   # probabilidad de 'bad'

    # Umbrales calibrados para desbalance 70/30
    if prob_default < 0.30:
        risk_level = 0
        risk_label = "Bajo riesgo"
    elif prob_default < 0.55:
        risk_level = 1
        risk_label = "Riesgo medio"
    else:
        risk_level = 2
        risk_label = "Alto riesgo"

    return {
        "risk_level": risk_level,
        "risk_label": risk_label,
        "probability_default": round(prob_default * 100, 2),
        "probabilities": {
            "good (no default)": round(float(proba[0]) * 100, 2),
            "bad (default)":     round(prob_default * 100, 2),
        },
    }


def predict_batch(df_raw: pd.DataFrame, model_name: str = "best_model") -> pd.DataFrame:
    """
    Predicción en lote sobre un DataFrame con columnas originales (sin encodear).
    """
    results = [predict(row.to_dict(), model_name=model_name)
               for _, row in df_raw.iterrows()]
    out = df_raw.copy()
    out["risk_level"]          = [r["risk_level"]          for r in results]
    out["risk_label"]          = [r["risk_label"]          for r in results]
    out["probability_default"] = [r["probability_default"] for r in results]
    return out
