# Análisis de Riesgo Crediticio — EAFIT IA 2026

Sistema de evaluación crediticia que combina un modelo ML (XGBoost) con generación de reportes narrativos mediante LLM (Llama 3 via Groq).

🚀 **App desplegada:** https://proyecto-ia-eafit-2026-1.streamlit.app/

## Estructura del proyecto

```
proyecto-ia-eafit/
├── app/
│   ├── main.py                  # Aplicación Streamlit
│   └── .streamlit/secrets.toml  # API keys (NO subir a GitHub)
├── data/
│   ├── raw/                     # Dataset original German Credit
│   └── processed/               # Datos preprocesados + encoders
├── models/
│   ├── checkpoints/             # Modelo entrenado (best_model.pkl)
│   └── plots/                   # Gráficas de evaluación
├── notebooks/
│   ├── 02_preprocessing.ipynb   # EDA y preprocesamiento
│   ├── 03_modeling.ipynb        # Entrenamiento ML (XGBoost ganador)
│   └── 04_llm_app.ipynb         # Prototipo LLM con Groq
├── src/
│   ├── models/train.py          # Entrenamiento de modelos
│   ├── models/predict.py        # Pipeline de predicción
│   ├── evaluation/metrics.py    # Métricas de evaluación
│   └── generator.py             # Generación de reportes con Groq
├── docs/                        # Informe final PDF
└── requirements.txt
```

## Instalación

```bash
git clone https://github.com/tomechava/proyecto-ia-eafit.git
cd proyecto-ia-eafit
pip install -r requirements.txt
```

## Configurar API Key de Groq

Crear el archivo `app/.streamlit/secrets.toml` con:

```toml
GROQ_API_KEY = "tu_api_key_aqui"
```

Obtener key gratuita en: https://console.groq.com/keys

## Ejecutar la app

```bash
streamlit run app/main.py
```

La app estará disponible en http://localhost:8501

## Notebooks

Ejecutar en orden desde la raíz del proyecto:

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
jupyter notebook notebooks/04_llm_app.ipynb
```

## Modelo

- **Algoritmo:** XGBoost con `scale_pos_weight=2.33` para desbalance 70/30
- **Métricas test:** Accuracy 0.73 · F1 0.55 · ROC-AUC 0.77
- **Dataset:** German Credit Risk (UCI) — 1000 clientes, 9 features
- **Umbrales de riesgo:** Bajo <30% · Medio 30-55% · Alto >55%

## Video demo

🎥 [*Video*](https://drive.google.com/file/d/10o8cH05EuHobrBq5OEQ61rkIbgDxR7ZO/view?usp=sharing)

## Equipo

- Tomas Echavarria — EDA y preprocesamiento
- Santiago Sanchez — Modelado ML
- Nathan Martinez — Integración LLM y app Streamlit
