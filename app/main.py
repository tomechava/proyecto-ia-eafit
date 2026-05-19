import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

from src.models.predict import predict
from src.generator import generate_credit_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Riesgo Crediticio · EAFIT IA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #003d79;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-card h4 { margin: 0 0 0.2rem 0; color: #003d79; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; color: #1a1510; }

    .risk-low    { background:#e6f4ed; border-left-color:#006b3c; }
    .risk-low h4 { color:#006b3c; }
    .risk-low p  { color:#006b3c; }

    .risk-mid    { background:#fef3e2; border-left-color:#b45309; }
    .risk-mid h4 { color:#b45309; }
    .risk-mid p  { color:#b45309; }

    .risk-high   { background:#fdf0f0; border-left-color:#9b1b1b; }
    .risk-high h4{ color:#9b1b1b; }
    .risk-high p { color:#9b1b1b; }

    .prompt-box {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: monospace;
        font-size: 0.82rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .tag {
        display: inline-block;
        background: #003d79;
        color: white;
        border-radius: 20px;
        padding: 0.15rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .tag.green { background: #006b3c; }
    .tag.amber { background: #b45309; }
    .tag.red   { background: #9b1b1b; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_meta():
    model = joblib.load("models/checkpoints/best_model.pkl")
    with open("models/checkpoints/model_metadata.json") as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_dataset():
    df = pd.read_csv("data/raw/german_credit_data.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df

model, meta = load_model_and_meta()
df_raw = load_dataset()

RISK_STYLE = {0: ("risk-low",  "✅", "Bajo riesgo",  "#006b3c"),
              1: ("risk-mid",  "⚠️", "Riesgo medio", "#b45309"),
              2: ("risk-high", "🚨", "Alto riesgo",  "#9b1b1b")}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Logo_EAFIT.svg/320px-Logo_EAFIT.svg.png", width=140)
    st.markdown("### 🏦 Riesgo Crediticio")
    st.caption("EAFIT · Inteligencia Artificial · 2026-1")
    st.divider()
    tab_sel = st.radio(
        "Navegación",
        ["🏠 Inicio", "🔍 Predicción", "📊 Rendimiento del Modelo", "📈 Análisis Exploratorio", "🤖 Arquitectura"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Modelo:** XGBoost · ROC-AUC 0.77")
    st.caption("**LLM:** Llama 3.1 8B via Groq")
    st.caption("**Dataset:** German Credit Risk (UCI) · n=1000")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INICIO
# ══════════════════════════════════════════════════════════════════════════════
if tab_sel == "🏠 Inicio":
    st.title("Sistema de Análisis de Riesgo Crediticio")
    st.markdown(
        "Pipeline completo de **Machine Learning + LLM** para evaluar solicitudes de crédito "
        "y generar reportes narrativos explicativos."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clientes en dataset", "1,000")
    col2.metric("Features utilizadas", "19")
    col3.metric("ROC-AUC (test)", "0.77")
    col4.metric("Modelos evaluados", "4")

    st.divider()

    st.subheader("Pipeline del sistema")
    st.markdown("""
    ```
    Datos crudos (German Credit UCI)
         │
         ▼
    [EDA + Preprocesamiento]  ──  Notebook 02
         │  OrdinalEncoder · StandardScaler · OHE
         ▼
    [Entrenamiento ML]  ──  Notebook 03
         │  Logistic Regression · Random Forest
         │  Gradient Boosting · XGBoost ← mejor
         ▼
    [Predicción en tiempo real]  ──  src/models/predict.py
         │  Probabilidad de incumplimiento
         │  Clasificación: Bajo / Medio / Alto riesgo
         ▼
    [Generación de Reporte LLM]  ──  src/generator.py
         │  Llama 3.1-8B via Groq API
         │  Prompt estructurado con perfil + resultado ML
         ▼
    [Interfaz Streamlit]  ──  app/main.py
    ```
    """)

    st.divider()
    st.subheader("Equipo de desarrollo")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Tomas Echavarria**")
        st.caption("EDA · Preprocesamiento · Pipeline de datos")
        st.markdown('<span class="tag">Notebook 02</span><span class="tag green">src/data</span>', unsafe_allow_html=True)
    with c2:
        st.markdown("**Santiago Sanchez**")
        st.caption("Modelado ML · Evaluación · Artefactos")
        st.markdown('<span class="tag">Notebook 03</span><span class="tag green">src/models</span>', unsafe_allow_html=True)
    with c3:
        st.markdown("**Nathan Martinez**")
        st.caption("Integración LLM · App Streamlit")
        st.markdown('<span class="tag">Notebook 04</span><span class="tag green">src/generator</span>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Dataset: German Credit Risk")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        El **German Credit Dataset** (UCI) contiene 1,000 solicitudes de crédito reales
        del sistema bancario alemán, con 9 features originales que describen el perfil financiero
        y personal del solicitante.

        **Variable objetivo:** `Risk`
        - `good` (bajo riesgo) → 700 casos (70%)
        - `bad` (alto riesgo)  → 300 casos (30%)

        **Desbalance de clases** manejado con `scale_pos_weight = 2.33` en XGBoost.
        """)
    with c2:
        fig = px.pie(
            values=[700, 300],
            names=["Good (bajo riesgo)", "Bad (alto riesgo)"],
            color_discrete_sequence=["#006b3c", "#9b1b1b"],
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=260,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif tab_sel == "🔍 Predicción":
    st.title("🔍 Predicción de Riesgo Crediticio")
    st.markdown("Ingresa el perfil del solicitante. El modelo ML clasifica el riesgo y el LLM genera un reporte explicativo.")
    st.divider()

    with st.form("prediction_form"):
        st.subheader("Perfil del solicitante")
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.slider("Edad", 18, 80, 35)
            sex    = st.selectbox("Sexo", ["male", "female"])
            job    = st.selectbox("Nivel de trabajo",
                        options=[0,1,2,3],
                        format_func=lambda x: {0:"0 – Sin calificar (no residente)",
                                               1:"1 – Sin calificar (residente)",
                                               2:"2 – Calificado",
                                               3:"3 – Altamente calificado"}[x], index=2)
        with c2:
            housing         = st.selectbox("Vivienda", ["own","free","rent"])
            saving_accounts = st.selectbox("Cuenta de ahorros",  ["little","moderate","quite rich","rich"])
            checking_account= st.selectbox("Cuenta corriente",   ["little","moderate","quite rich","rich"])
        with c3:
            credit_amount = st.number_input("Monto solicitado (USD)", 500, 100_000, 5_000, 500)
            duration      = st.slider("Duración (meses)", 6, 72, 24)
            purpose       = st.selectbox("Propósito", ["car","furniture/equipment","radio/TV",
                                          "domestic appliances","repairs","education",
                                          "business","vacation/others"])

        submitted = st.form_submit_button("Analizar solicitud", type="primary", use_container_width=True)

    if not submitted:
        st.stop()

    input_data = {
        "Age": age, "Sex": sex, "Job": job, "Housing": housing,
        "Saving accounts": saving_accounts, "Checking account": checking_account,
        "Credit amount": credit_amount, "Duration": duration, "Purpose": purpose,
    }

    # ── ML prediction ─────────────────────────────────────────────────────────
    with st.spinner("Ejecutando modelo XGBoost..."):
        result     = predict(input_data)
        risk_level = result["risk_level"]
        prob_bad   = result["probability_default"]
        prob_good  = result["probabilities"]["good (no default)"]
        css_cls, icon, label, color = RISK_STYLE[risk_level]

    st.divider()
    st.subheader("Resultado del Modelo ML")

    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="metric-card {css_cls}"><h4>Clasificación de riesgo</h4><p>{icon} {label}</p></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card {css_cls}"><h4>Probabilidad de incumplimiento</h4><p>{prob_bad}%</p></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><h4>Probabilidad de pago</h4><p>{prob_good}%</p></div>', unsafe_allow_html=True)

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_bad,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Probabilidad de Incumplimiento", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  30], "color": "#e6f4ed"},
                {"range": [30, 55], "color": "#fef3e2"},
                {"range": [55,100], "color": "#fdf0f0"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": prob_bad},
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))

    # Donut probabilities
    fig_donut = go.Figure(go.Pie(
        labels=["Pago (good)", "Incumplimiento (bad)"],
        values=[prob_good, prob_bad],
        hole=0.55,
        marker_colors=["#006b3c", "#9b1b1b"],
        textinfo="label+percent",
    ))
    fig_donut.update_layout(height=280, margin=dict(t=20, b=10, l=10, r=10),
                            showlegend=False)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_gauge,  use_container_width=True)
    with g2:
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Feature contribution mini-bar ────────────────────────────────────────
    st.subheader("Variables más influyentes en esta predicción")
    features   = meta["feature_names"]
    importances= model.feature_importances_
    fi_df = pd.DataFrame({"Feature": features, "Importancia": importances})\
              .sort_values("Importancia", ascending=False).head(8)
    fig_fi = px.bar(fi_df, x="Importancia", y="Feature", orientation="h",
                    color="Importancia", color_continuous_scale="Blues",
                    title="Top 8 Features por Importancia (modelo global)")
    fig_fi.update_layout(height=320, yaxis=dict(autorange="reversed"),
                         margin=dict(t=40, b=10, l=10, r=10),
                         coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── LLM Report ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Reporte del Analista (Llama 3.1 via Groq)")
    st.caption("Generado automáticamente a partir del perfil del cliente y el resultado del modelo ML.")

    with st.spinner("Generando reporte con Llama 3.1-8B..."):
        try:
            report = generate_credit_report(input_data, label, prob_bad)
            st.markdown(f">{report.replace(chr(10), '  \n>')}")
        except Exception as e:
            st.warning(f"No se pudo generar el reporte LLM: {e}")

    with st.expander("🔧 Ver datos técnicos completos"):
        st.json(result)
        st.json(input_data)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RENDIMIENTO DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif tab_sel == "📊 Rendimiento del Modelo":
    st.title("📊 Evaluación y Rendimiento del Modelo")
    st.markdown("Comparación de los 4 modelos entrenados y métricas de evaluación del mejor modelo (XGBoost).")
    st.divider()

    # ── Model comparison table ────────────────────────────────────────────────
    st.subheader("Comparación de modelos entrenados")
    comparison_data = {
        "Modelo":         ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost ⭐"],
        "CV ROC-AUC":     ["0.748 ± 0.031",       "0.761 ± 0.028",  "0.771 ± 0.025",    "0.783 ± 0.022"],
        "Val Accuracy":   [0.700,                  0.720,            0.730,               0.740],
        "Val F1 Macro":   [0.510,                  0.530,            0.540,               0.548],
        "Val ROC-AUC":    [0.741,                  0.754,            0.762,               0.775],
        "Test ROC-AUC":   [0.721,                  0.739,            0.751,               0.767],
    }
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(
        df_comp.style
            .highlight_max(subset=["Val Accuracy","Val F1 Macro","Val ROC-AUC","Test ROC-AUC"],
                           color="#d4edda")
            .format({"Val Accuracy": "{:.3f}", "Val F1 Macro": "{:.3f}",
                     "Val ROC-AUC": "{:.3f}", "Test ROC-AUC": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart comparison
    fig_comp = go.Figure()
    metrics  = ["Val Accuracy", "Val F1 Macro", "Val ROC-AUC"]
    colors   = ["#5b9bd5", "#70ad47", "#ed7d31"]
    for m, c in zip(metrics, colors):
        fig_comp.add_trace(go.Bar(name=m, x=df_comp["Modelo"], y=df_comp[m],
                                  marker_color=c))
    fig_comp.update_layout(barmode="group", height=360,
                           title="Comparación de métricas por modelo",
                           yaxis=dict(range=[0.45, 0.82]),
                           margin=dict(t=50, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # ── Test metrics ──────────────────────────────────────────────────────────
    st.subheader("Métricas del mejor modelo en test (XGBoost)")
    m = meta["test_metrics"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['accuracy']:.4f}")
    c2.metric("Precision", f"{m['precision']:.4f}")
    c3.metric("Recall",    f"{m['recall']:.4f}")
    c4.metric("F1-Score",  f"{m['f1']:.4f}")
    c5.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")

    st.caption("""
    **Nota sobre métricas:** Con desbalance 70/30, la Accuracy sola es engañosa.
    Priorizamos **ROC-AUC** (discriminación global) y **F1 Macro** (balance entre clases).
    El umbral de clasificación se calibró en 0.30/0.55 para reducir falsos negativos en Alto Riesgo.
    """)

    st.divider()

    # ── Plots from training ───────────────────────────────────────────────────
    st.subheader("Visualizaciones de evaluación")

    tab_cm, tab_roc, tab_pr, tab_fi, tab_baseline = st.tabs(
        ["Matriz de Confusión", "Curva ROC", "Curva PR", "Feature Importance", "Baseline vs XGBoost"]
    )

    with tab_cm:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Baseline (Logistic Regression)**")
            img = Image.open("models/plots/cm_baseline.png")
            st.image(img, use_container_width=True)
        with c2:
            st.markdown("**XGBoost (mejor modelo)**")
            img = Image.open("models/plots/cm_xgboost.png")
            st.image(img, use_container_width=True)
        st.caption("La matriz de confusión muestra cómo XGBoost mejora la detección de la clase 'bad' (alto riesgo) respecto al baseline.")

    with tab_roc:
        img = Image.open("models/plots/roc_xgboost.png")
        st.image(img, use_container_width=True)
        st.caption(f"ROC-AUC = {m['roc_auc']:.4f} — El área bajo la curva mide la capacidad discriminativa del modelo en todos los umbrales posibles.")

    with tab_pr:
        img = Image.open("models/plots/pr_xgboost.png")
        st.image(img, use_container_width=True)
        st.caption("Curva Precision-Recall — Especialmente relevante con desbalance de clases. Muestra el trade-off entre precisión y recall para la clase 'bad'.")

    with tab_fi:
        img = Image.open("models/plots/fi_xgboost.png")
        st.image(img, use_container_width=True)
        # Interactive version
        features_all = meta["feature_names"]
        imp_all      = model.feature_importances_
        fi_df_all = pd.DataFrame({"Feature": features_all, "Importancia": imp_all})\
                      .sort_values("Importancia", ascending=True)
        fig_fi_all = px.bar(fi_df_all, x="Importancia", y="Feature", orientation="h",
                            color="Importancia", color_continuous_scale="Blues",
                            title="Feature Importance — XGBoost (interactivo)")
        fig_fi_all.update_layout(height=500, coloraxis_showscale=False,
                                 margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_fi_all, use_container_width=True)
        st.caption("**Checking account** es la variable más predictiva: refleja la liquidez inmediata del solicitante.")

    with tab_baseline:
        img = Image.open("models/plots/model_comparison.png")
        st.image(img, use_container_width=True)
        st.caption("Comparación visual de todos los modelos entrenados. XGBoost supera consistentemente al baseline en ROC-AUC.")

    st.divider()
    st.subheader("Estrategia de manejo de desbalance")
    st.markdown(f"""
    | Parámetro | Valor | Justificación |
    |-----------|-------|---------------|
    | `scale_pos_weight` | **2.33** | Ratio 70/30 → da más peso a la clase `bad` |
    | Umbral bajo riesgo | **< 30%** | Reduce falsos negativos en créditos seguros |
    | Umbral medio riesgo | **30–55%** | Zona de revisión manual por analista |
    | Umbral alto riesgo | **> 55%** | Rechazar automáticamente |
    | Métrica principal | **ROC-AUC** | Robusto ante desbalance, no depende del umbral |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif tab_sel == "📈 Análisis Exploratorio":
    st.title("📈 Análisis Exploratorio de Datos (EDA)")
    st.markdown(f"Dataset: **German Credit Risk (UCI)** · {len(df_raw):,} solicitudes · {df_raw.shape[1]} variables originales")
    st.divider()

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.subheader("Resumen estadístico")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total clientes",   f"{len(df_raw):,}")
    c2.metric("Buenos pagadores", f"{(df_raw['Risk']=='good').sum():,} (70%)")
    c3.metric("Morosos",          f"{(df_raw['Risk']=='bad').sum():,} (30%)")
    c4.metric("Features",         f"{df_raw.shape[1]-1}")

    with st.expander("Ver estadísticas descriptivas"):
        st.dataframe(df_raw.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.divider()

    # ── EDA plots from notebooks ───────────────────────────────────────────────
    st.subheader("Visualizaciones del EDA (generadas en Notebook 02)")
    plots_dir = "data/processed/plots"
    plot_files = {
        "Distribución de la variable objetivo":  "01_target_distribution.png",
        "Distribuciones numéricas":              "02_numeric_distributions.png",
        "Variables categóricas vs Riesgo":       "03_categorical_vs_risk.png",
        "Matriz de correlación":                 "04_correlation_matrix.png",
    }
    for title, fname in plot_files.items():
        path = os.path.join(plots_dir, fname)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(Image.open(path), use_container_width=True)
            st.markdown("")

    st.divider()

    # ── Interactive EDA ────────────────────────────────────────────────────────
    st.subheader("Exploración interactiva")

    tab_dist, tab_cat, tab_scatter, tab_null = st.tabs(
        ["Distribuciones numéricas", "Categóricas vs Riesgo", "Dispersión", "Valores nulos"]
    )

    with tab_dist:
        num_col = st.selectbox("Variable numérica", ["Age", "Credit amount", "Duration"])
        fig = px.histogram(df_raw, x=num_col, color="Risk",
                           color_discrete_map={"good":"#006b3c","bad":"#9b1b1b"},
                           barmode="overlay", nbins=30, opacity=0.75,
                           title=f"Distribución de {num_col} por Riesgo")
        fig.update_layout(height=380, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_cat:
        cat_col = st.selectbox("Variable categórica",
                               ["Purpose","Housing","Sex","Saving accounts","Checking account"])
        ct = df_raw.groupby([cat_col, "Risk"]).size().reset_index(name="count")
        fig = px.bar(ct, x=cat_col, y="count", color="Risk",
                     color_discrete_map={"good":"#006b3c","bad":"#9b1b1b"},
                     barmode="group",
                     title=f"{cat_col} vs Riesgo crediticio")
        fig.update_layout(height=380, margin=dict(t=50,b=10,l=10,r=10),
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with tab_scatter:
        col_x = st.selectbox("Eje X", ["Age","Credit amount","Duration"], index=1)
        col_y = st.selectbox("Eje Y", ["Duration","Age","Credit amount"], index=0)
        fig = px.scatter(df_raw, x=col_x, y=col_y, color="Risk",
                         color_discrete_map={"good":"#006b3c","bad":"#9b1b1b"},
                         opacity=0.6, title=f"{col_x} vs {col_y}")
        fig.update_layout(height=400, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_null:
        null_counts = df_raw.isnull().sum().reset_index()
        null_counts.columns = ["Feature", "Nulos"]
        null_counts["% Nulos"] = (null_counts["Nulos"] / len(df_raw) * 100).round(2)
        st.dataframe(null_counts, use_container_width=True, hide_index=True)
        total_nulos = null_counts["Nulos"].sum()
        if total_nulos == 0:
            st.success("✅ El dataset no tiene valores nulos — no se requirió imputación.")
        else:
            st.warning(f"⚠️ Se encontraron {total_nulos} valores nulos. Ver Notebook 02 para la estrategia de manejo.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ARQUITECTURA
# ══════════════════════════════════════════════════════════════════════════════
elif tab_sel == "🤖 Arquitectura":
    st.title("🤖 Arquitectura del Sistema")
    st.markdown("Documentación técnica del pipeline, los prompts utilizados y las decisiones de diseño.")
    st.divider()

    st.subheader("Componentes del sistema")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Componente ML (src/models/)**
        - `train.py` — entrenamiento de 4 modelos con CV
        - `predict.py` — pipeline de predicción en producción
        - `best_model.pkl` — XGBoost serializado con joblib

        **Preprocesamiento (data/processed/)**
        - `scaler.pkl` — StandardScaler (Age, Credit amount, Duration)
        - `ordinal_encoder.pkl` — OrdinalEncoder (Saving/Checking accounts)
        - `feature_columns.json` — orden de columnas para alineación
        """)
    with c2:
        st.markdown("""
        **Componente LLM (src/generator.py)**
        - Modelo: `llama-3.1-8b-instant` via Groq API
        - Temperatura: `0.4` (respuestas consistentes pero no rígidas)
        - Max tokens: `600`
        - Rol del sistema: analista de riesgo crediticio bancario

        **Interfaz (app/main.py)**
        - Framework: Streamlit
        - Gráficas interactivas: Plotly
        - Secretos: `.streamlit/secrets.toml` (no en repo)
        """)

    st.divider()
    st.subheader("Prompts documentados")
    st.caption("Los prompts son parte de la arquitectura del sistema y están justificados a continuación.")

    with st.expander("📝 System Prompt — Rol del analista LLM", expanded=True):
        st.markdown('<div class="prompt-box">Eres un asistente experto en análisis de riesgo crediticio para entidades bancarias.\nTus reportes son profesionales, concisos y basados en evidencia cuantitativa.</div>', unsafe_allow_html=True)
        st.markdown("""
        **Justificación:** El rol de sistema establece el dominio de expertise del modelo.
        Al declarar "basados en evidencia cuantitativa" se reduce la tendencia del LLM a generar
        afirmaciones no fundamentadas en los datos del cliente.
        """)

    with st.expander("📝 User Prompt — Generación del reporte crediticio"):
        st.markdown("""
        ```
        Actúa como un analista de riesgo financiero experto de una entidad bancaria latinoamericana.
        Un cliente solicita un crédito con el siguiente perfil:

        - Edad: {Age} años
        - Sexo: {Sex}
        - Nivel de trabajo: {Job} (0=sin calificar, 1=no calificado, 2=calificado, 3=altamente calificado)
        - Tipo de vivienda: {Housing}
        - Cuenta de ahorros: {Saving accounts}
        - Cuenta corriente: {Checking account}
        - Monto solicitado: ${Credit amount} USD
        - Duración del crédito: {Duration} meses
        - Propósito del crédito: {Purpose}

        Nuestro modelo predictivo (XGBoost, ROC-AUC = 0.77) ha clasificado a este cliente
        como **{resultado_ml}** con una probabilidad de incumplimiento del {probabilidad}%.

        Redacta un reporte profesional de 2 a 3 párrafos en español que:
        1. Explique los factores del perfil que más influyen en esta clasificación.
        2. Justifique la decisión del modelo de forma clara para el cliente y la entidad.
        3. Proporcione una recomendación concreta: aprobación, aprobación con condiciones,
           o rechazo del crédito.
        ```
        """)
        st.markdown("""
        **Justificación de decisiones en el prompt:**

        | Decisión | Razón |
        |----------|-------|
        | Incluir ROC-AUC del modelo | El LLM puede calibrar su confianza según la precisión real del modelo |
        | Temperatura 0.4 | Respuestas profesionales y reproducibles, sin ser mecánicas |
        | 3 párrafos estructurados | Guía al modelo hacia un reporte accionable, no una lista |
        | Contexto latinoamericano | Adapta el lenguaje y las referencias culturales del reporte |
        | Proporcionar probabilidad exacta | Ancla el LLM a la salida cuantitativa del modelo, evitando alucinaciones |
        """)

    st.divider()
    st.subheader("Decisiones técnicas clave")
    st.markdown("""
    | Decisión | Alternativa considerada | Razón de la elección |
    |----------|------------------------|----------------------|
    | XGBoost como modelo final | Random Forest, Gradient Boosting | Mejor ROC-AUC en validación (0.775) y manejo nativo de desbalance con `scale_pos_weight` |
    | Groq API (Llama 3.1) | OpenAI GPT-4, Ollama local | Gratuito, baja latencia (<1s), sin instalación local |
    | OrdinalEncoder para cuentas | OneHotEncoding | Las categorías tienen orden inherente (little < moderate < quite rich < rich) |
    | StandardScaler para numéricas | MinMaxScaler | Más robusto ante outliers en `Credit amount` |
    | Umbral 0.30/0.55 (no 0.5) | Umbral único 0.5 | Dataset desbalanceado: umbral 0.5 sesga hacia la clase mayoritaria |
    | Split estratificado 70/15/15 | Split aleatorio | Preserva la proporción 70/30 de clases en train/val/test |
    """)

    st.divider()
    st.subheader("Estructura del repositorio")
    st.code("""
proyecto-ia-eafit/
├── app/
│   ├── main.py                       # Interfaz Streamlit (este archivo)
│   └── .streamlit/secrets.toml       # API keys — NO en GitHub
├── data/
│   ├── raw/german_credit_data.csv    # Dataset original UCI
│   └── processed/                    # Encoders, scaler, splits, plots EDA
├── models/
│   ├── checkpoints/
│   │   ├── best_model.pkl            # XGBoost serializado
│   │   └── model_metadata.json       # Métricas y configuración
│   └── plots/                        # Confusion matrix, ROC, PR, FI
├── notebooks/
│   ├── 02_preprocessing.ipynb        # EDA y preprocesamiento (Tomas)
│   ├── 03_modeling.ipynb             # Entrenamiento ML (Santiago)
│   └── 04_llm_app.ipynb              # Prototipo LLM (Nathan)
├── src/
│   ├── models/train.py               # Entrenamiento con CV
│   ├── models/predict.py             # Pipeline de predicción
│   ├── evaluation/metrics.py         # Métricas y visualizaciones
│   └── generator.py                  # Generación de reportes LLM
├── docs/                             # Informe final PDF
├── requirements.txt
└── README.md
    """, language="")
