# Guia de Usuario
## Sistema de Analisis de Riesgo Crediticio
**EAFIT · Inteligencia Artificial · 2026-1**

App desplegada: https://proyecto-ia-eafit-2026-1.streamlit.app

---

## Tabla de Contenidos

1. [Requisitos e Instalacion](#1-requisitos-e-instalacion)
2. [Ejecucion local](#2-ejecucion-local)
3. [Navegacion por la app](#3-navegacion-por-la-app)
4. [Seccion Inicio](#4-seccion-inicio)
5. [Seccion Prediccion](#5-seccion-prediccion)
6. [Seccion Rendimiento del Modelo](#6-seccion-rendimiento-del-modelo)
7. [Seccion Analisis Exploratorio](#7-seccion-analisis-exploratorio)
8. [Seccion Arquitectura](#8-seccion-arquitectura)
9. [Preguntas frecuentes](#9-preguntas-frecuentes)

---

## 1. Requisitos e Instalacion

**Requisitos del sistema:**
- Python 3.10 o superior
- pip actualizado
- Cuenta gratuita en [console.groq.com](https://console.groq.com) para obtener una API key

**Pasos de instalacion:**

```bash
# 1. Clonar el repositorio
git clone https://github.com/tomechava/proyecto-ia-eafit.git
cd proyecto-ia-eafit

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Crear el archivo de secretos
mkdir -p app/.streamlit
echo 'GROQ_API_KEY = "tu_api_key_aqui"' > app/.streamlit/secrets.toml
```

Las dependencias principales son:

| Paquete | Version recomendada | Uso |
|---------|-------------------|-----|
| streamlit | >= 1.35 | Interfaz de usuario |
| xgboost | >= 2.0 | Modelo de clasificacion |
| groq | >= 0.9 | API de Llama 3.1 |
| plotly | >= 5.0 | Graficas interactivas |
| scikit-learn | >= 1.4 | Preprocesamiento |
| pandas | >= 2.0 | Manejo de datos |

---

## 2. Ejecucion local

Desde la raiz del proyecto ejecutar:

```bash
streamlit run app/main.py
```

La app abre automaticamente en el navegador en `http://localhost:8501`.

> **Nota:** El comando debe ejecutarse desde la carpeta raiz del proyecto (`proyecto-ia-eafit/`), no desde dentro de `app/`. Esto garantiza que las rutas relativas a `data/` y `models/` funcionen correctamente.

---

## 3. Navegacion por la app

La app tiene un panel de navegacion en la barra lateral izquierda con cinco secciones:

```
🏠 Inicio
🔍 Prediccion
📊 Rendimiento del Modelo
📈 Analisis Exploratorio
🤖 Arquitectura
```

Haz click en cualquier opcion del panel para cambiar de seccion. La seccion activa se resalta en azul.

En la parte inferior del panel lateral siempre se muestran los datos tecnicos del sistema:
- Modelo: XGBoost con ROC-AUC 0.77
- LLM: Llama 3.1 8B via Groq
- Dataset: German Credit Risk (UCI), 1000 clientes

---

## 4. Seccion Inicio

Esta seccion presenta una vision general del proyecto.

**Metricas destacadas en la parte superior:**

| Tarjeta | Valor | Significado |
|---------|-------|-------------|
| Clientes en dataset | 1,000 | Total de solicitudes en el dataset de entrenamiento |
| Features utilizadas | 19 | Variables tras el proceso de encoding |
| ROC-AUC (test) | 0.77 | Capacidad discriminativa del modelo en datos no vistos |
| Modelos evaluados | 4 | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |

**Pipeline del sistema:** Diagrama de texto que muestra el flujo completo desde los datos crudos hasta la interfaz, pasando por EDA, preprocesamiento, entrenamiento ML y generacion LLM.

**Equipo de desarrollo:** Nombres y roles de cada integrante del equipo con los modulos que desarrollaron.

**Dataset German Credit Risk:** Descripcion del dataset con un grafico de dona interactivo que muestra la distribucion de clases (70% buenos pagadores, 30% morosos).

---

## 5. Seccion Prediccion

Esta es la seccion principal de uso. Permite evaluar el perfil de un solicitante de credito en tiempo real.

### 5.1 Formulario de entrada

El formulario tiene nueve campos organizados en tres columnas:

**Columna izquierda:**
- **Edad** (slider, 18 a 80 anos): Edad del solicitante en anos
- **Sexo** (selector): `male` o `female`
- **Nivel de trabajo** (selector con descripcion):
  - 0: Sin calificar, no residente
  - 1: Sin calificar, residente
  - 2: Calificado (valor por defecto)
  - 3: Altamente calificado

**Columna central:**
- **Vivienda** (selector): `own`, `free` o `rent`
- **Cuenta de ahorros** (selector): `little`, `moderate`, `quite rich` o `rich`
- **Cuenta corriente** (selector): `little`, `moderate`, `quite rich` o `rich`

**Columna derecha:**
- **Monto solicitado** (numero, 500 a 100,000 USD): Valor del credito en dolares
- **Duracion** (slider, 6 a 72 meses): Plazo de pago del credito
- **Proposito** (selector): car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others

Una vez completado el formulario, hacer click en el boton **"Analizar solicitud"** (azul, ancho completo).

### 5.2 Resultado del modelo ML

Tras hacer click en el boton aparecen tres tarjetas de resultado con codigo de color:

| Color | Clasificacion | Rango de probabilidad |
|-------|--------------|----------------------|
| Verde | Bajo riesgo | Menos del 30% |
| Naranja | Riesgo medio | Entre 30% y 55% |
| Rojo | Alto riesgo | Mas del 55% |

Las tarjetas muestran:
1. La clasificacion de riesgo con icono (checkmark, advertencia o alerta)
2. La probabilidad exacta de incumplimiento en porcentaje
3. La probabilidad de pago (complemento)

### 5.3 Visualizaciones interactivas

**Medidor de riesgo (gauge):** Semicirculo con tres zonas coloreadas (verde, naranja, rojo) que indica visualmente la posicion del solicitante en el espectro de riesgo. El puntero senala la probabilidad exacta.

**Grafico de dona:** Muestra la distribucion de probabilidades entre las dos clases (pago vs incumplimiento) en porcentajes.

**Top 8 variables influyentes:** Grafico de barras horizontales que muestra las ocho features con mayor importancia global en el modelo XGBoost. Permite al analista entender cuales factores del sistema son mas determinantes para cualquier prediccion.

### 5.4 Reporte narrativo del LLM

Bajo las visualizaciones aparece el reporte generado automaticamente por Llama 3.1 via Groq. El reporte:
- Explica los factores del perfil que mas influyen en la decision
- Justifica la clasificacion en lenguaje profesional
- Proporciona una recomendacion concreta (aprobacion, aprobacion con condiciones o rechazo)

> **Tiempo de respuesta:** El reporte tarda entre 3 y 8 segundos dependiendo de la carga del servidor de Groq. Un indicador de carga ("Generando reporte...") aparece mientras se procesa.

**Detalles tecnicos:** Al final de la pagina hay un panel expandible que muestra el JSON completo con todos los valores numericos de la prediccion y los datos de entrada enviados al modelo.

---

## 6. Seccion Rendimiento del Modelo

Documenta el proceso de evaluacion y los resultados del entrenamiento.

### 6.1 Tabla comparativa de modelos

Tabla con los cuatro modelos entrenados y sus metricas en los conjuntos de validacion y prueba. La fila del mejor modelo (XGBoost) aparece resaltada en verde. Las columnas son:

| Columna | Descripcion |
|---------|-------------|
| CV ROC-AUC | Media y desviacion estandar de la validacion cruzada de 5 pliegues |
| Val Accuracy | Exactitud en el conjunto de validacion |
| Val F1 Macro | F1 promedio entre clases en validacion |
| Val ROC-AUC | Area bajo la curva ROC en validacion |
| Test ROC-AUC | Area bajo la curva ROC en el conjunto de prueba no visto |

El grafico de barras agrupadas debajo muestra visualmente la comparacion entre modelos para las tres metricas principales.

### 6.2 Metricas del modelo final en prueba

Cinco tarjetas con las metricas exactas de XGBoost sobre el conjunto de prueba (150 instancias, nunca vistas durante el entrenamiento):

- **Accuracy:** 0.7333
- **Precision:** 0.5581
- **Recall:** 0.5333
- **F1-Score:** 0.5455
- **ROC-AUC:** 0.7672

### 6.3 Visualizaciones del entrenamiento

Cuatro pestanas con las graficas generadas durante el entrenamiento:

**Matriz de Confusion:** Compara la matriz del baseline (Regresion Logistica) con la de XGBoost, mostrando como el modelo final mejora la deteccion de la clase de alto riesgo.

**Curva ROC:** Muestra el tradeoff entre tasa de verdaderos positivos y falsos positivos en todos los umbrales. El area bajo la curva (AUC = 0.767) indica buena capacidad discriminativa.

**Curva PR:** Curva de Precision-Recall, especialmente relevante para evaluar el rendimiento en la clase minoritaria (morosos, 30% del dataset).

**Feature Importance:** Imagen estatica del entrenamiento mas version interactiva de Plotly con las 19 features ordenadas por importancia. Al pasar el cursor sobre cada barra se muestra el valor exacto.

**Baseline vs XGBoost:** Grafico comparativo generado al final del entrenamiento mostrando todos los modelos.

### 6.4 Estrategia de desbalance

Tabla que documenta las decisiones tecnicas para manejar el desbalance de clases 70/30, incluyendo el parametro `scale_pos_weight = 2.33` y la justificacion de los umbrales de clasificacion.

---

## 7. Seccion Analisis Exploratorio

Presenta los hallazgos del EDA realizado en el Notebook 02.

### 7.1 Resumen estadistico

Cuatro metricas en la parte superior: total de clientes, buenos pagadores, morosos y numero de features. Un panel expandible muestra la tabla completa de estadisticas descriptivas (media, desviacion estandar, percentiles) para todas las variables.

### 7.2 Graficas del notebook

Cuatro visualizaciones generadas durante el analisis exploratorio original:
1. Distribucion de la variable objetivo
2. Distribuciones de variables numericas
3. Variables categoricas vs riesgo
4. Matriz de correlaciones

### 7.3 Exploracion interactiva

Cuatro pestanas con graficas dinamicas:

**Distribuciones numericas:** Seleccionar Age, Credit amount o Duration para ver su histograma superpuesto por clase de riesgo (verde para buenos pagadores, rojo para morosos).

**Categoricas vs Riesgo:** Seleccionar Purpose, Housing, Sex, Saving accounts o Checking account para ver graficas de barras agrupadas que muestran la distribucion de riesgo dentro de cada categoria.

**Dispersion:** Cruzar dos variables numericas en un scatter plot coloreado por clase de riesgo. Util para identificar patrones bivariados.

**Valores nulos:** Tabla con el conteo y porcentaje de valores faltantes por variable. El dataset tiene valores nulos en Saving accounts (18.3%) y Checking account (39.4%), tratados como categoria adicional durante el preprocesamiento.

---

## 8. Seccion Arquitectura

Documentacion tecnica del sistema para usuarios avanzados.

### 8.1 Componentes

Descripcion de todos los modulos del sistema con rutas de archivos:
- `src/models/train.py`: entrenamiento
- `src/models/predict.py`: prediccion en produccion
- `src/generator.py`: integracion con Groq
- `app/main.py`: interfaz Streamlit

### 8.2 Prompts documentados

Dos paneles expandibles que muestran el system prompt y el user prompt exactos enviados al LLM, con tabla de justificacion para cada decision de diseno del prompt (temperatura, estructura, contexto incluido).

### 8.3 Decisiones tecnicas

Tabla comparativa que documenta cada decision de diseno del sistema frente a las alternativas consideradas, con la razon de la eleccion. Por ejemplo: por que XGBoost sobre Random Forest, por que Groq sobre OpenAI, por que OrdinalEncoder sobre OneHotEncoding para las cuentas.

### 8.4 Estructura del repositorio

Arbol completo del repositorio con descripcion de cada archivo y carpeta.

---

## 9. Preguntas frecuentes

**La app no carga o muestra error de conexion.**
Verificar que la API key de Groq en `app/.streamlit/secrets.toml` sea valida. Las keys gratuitas de Groq tienen limite de solicitudes por minuto; esperar unos segundos y reintentar.

**El reporte LLM no aparece y muestra un mensaje de error.**
El modelo `llama-3.1-8b-instant` puede estar temporalmente saturado en el servidor de Groq. El sistema muestra el resultado del modelo ML de todas formas; el reporte LLM es un componente adicional.

**Los graficos no se ven o aparecen en blanco.**
Asegurar que plotly este instalado con `pip install plotly`. Si se ejecuta en un entorno virtual, activarlo antes de lanzar la app.

**Al ejecutar localmente aparece un error de ruta de archivos.**
Verificar que el comando `streamlit run app/main.py` se ejecuta desde la carpeta raiz del proyecto, no desde dentro de `app/`.

**Los modelos PKL no cargan.**
Los archivos `models/checkpoints/best_model.pkl`, `data/processed/scaler.pkl`, `data/processed/ordinal_encoder.pkl` y `data/processed/feature_columns.json` deben estar presentes. Si se clono el repo recientemente y no estan, ejecutar el Notebook 03 para regenerarlos.

---

*Guia elaborada por el equipo del proyecto. EAFIT · Inteligencia Artificial · 2026-1*
