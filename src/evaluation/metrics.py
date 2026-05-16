import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    average_precision_score,
)


def evaluate_model(model, X, y, model_name="model") -> dict:
    """
    Evalúa un modelo y retorna dict con todas las métricas clave.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # ROC-AUC: maneja binario y multiclase
    if y_proba.shape[1] == 2:
        roc_auc = roc_auc_score(y, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")

    metrics = {
        "val_accuracy": round(accuracy_score(y, y_pred), 4),
        "val_f1_macro": round(f1_score(y, y_pred, average="macro"), 4),
        "val_roc_auc": round(roc_auc, 4),
    }

    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {metrics['val_accuracy']}")
    print(f"F1 Macro : {metrics['val_f1_macro']}")
    print(f"ROC-AUC  : {metrics['val_roc_auc']}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    return metrics


def plot_confusion_matrix(model, X, y, model_name="model", save_path=None):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_roc_curve(model, X, y, model_name="model", save_path=None):
    """Solo para clasificación binaria."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title(f"ROC Curve — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_precision_recall(model, X, y, model_name="model", save_path=None):
    """Solo para clasificación binaria."""
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_proba)
    ap = average_precision_score(y, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def compare_models(results: dict) -> pd.DataFrame:
    """
    Genera tabla comparativa de todos los modelos entrenados.
    
    Args:
        results: dict retornado por train_all()
    
    Returns:
        DataFrame con métricas ordenadas por ROC-AUC desc.
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Modelo": name,
            "CV ROC-AUC": f"{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}",
            "Val Accuracy": r["val_accuracy"],
            "Val F1 Macro": r["val_f1_macro"],
            "Val ROC-AUC": r["val_roc_auc"],
        })
    df = pd.DataFrame(rows).sort_values("Val ROC-AUC", ascending=False).reset_index(drop=True)
    print("\n=== Comparación de Modelos ===")
    print(df.to_string(index=False))
    return df
