import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from src.evaluation.metrics import evaluate_model

CHECKPOINTS_DIR = "models/checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def get_models():
    """
    Todos los modelos con class_weight='balanced' para manejar el desbalance
    70% good / 30% bad del German Credit dataset.
    XGBoost usa scale_pos_weight en su lugar.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=2.33,   # aprox 70/30 → peso a la clase minoritaria 'bad'
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def train_all(X_train, y_train, X_val, y_val):
    """Entrena todos los modelos y devuelve resultados comparativos."""
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Entrenando: {name}")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        val_metrics = evaluate_model(model, X_val, y_val, model_name=name)

        results[name] = {
            "model": model,
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
            **val_metrics,
        }

        print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        save_model(model, name)

    return results


def save_model(model, name):
    path = os.path.join(CHECKPOINTS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def load_model(name):
    path = os.path.join(CHECKPOINTS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el modelo en: {path}")
    return joblib.load(path)


def select_best_model(results, metric="val_roc_auc"):
    best_name = max(results, key=lambda k: results[k].get(metric, 0))
    print(f"\nMejor modelo: {best_name} ({metric}={results[best_name][metric]:.4f})")
    return best_name, results[best_name]["model"]
