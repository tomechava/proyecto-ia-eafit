import streamlit as st
from groq import Groq


def generate_credit_report(datos_cliente: dict, resultado_ml: str, probabilidad: float) -> str:
    """
    Genera un reporte narrativo de riesgo crediticio usando Llama 3 via Groq.

    Args:
        datos_cliente: dict con los campos originales del cliente.
        resultado_ml:  etiqueta de riesgo del modelo ("Bajo riesgo", "Riesgo medio", "Alto riesgo").
        probabilidad:  probabilidad de incumplimiento en % (0-100).

    Returns:
        Reporte en texto plano (2-3 párrafos).
    """
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Actúa como un analista de riesgo financiero experto de una entidad bancaria latinoamericana.
Un cliente solicita un crédito con el siguiente perfil:

- Edad: {datos_cliente['Age']} años
- Sexo: {datos_cliente['Sex']}
- Nivel de trabajo: {datos_cliente['Job']} (0=sin calificar, 1=no calificado, 2=calificado, 3=altamente calificado)
- Tipo de vivienda: {datos_cliente['Housing']}
- Cuenta de ahorros: {datos_cliente['Saving accounts']}
- Cuenta corriente: {datos_cliente['Checking account']}
- Monto solicitado: ${datos_cliente['Credit amount']:,} USD
- Duración del crédito: {datos_cliente['Duration']} meses
- Propósito del crédito: {datos_cliente['Purpose']}

Nuestro modelo predictivo (XGBoost, ROC-AUC = 0.77, entrenado sobre German Credit Dataset) ha clasificado a este cliente como **{resultado_ml}** con una probabilidad de incumplimiento del {probabilidad:.1f}%.

Redacta un reporte profesional de 2 a 3 párrafos en español que:
1. Explique los factores del perfil que más influyen en esta clasificación.
2. Justifique la decisión del modelo de forma clara para el cliente y la entidad.
3. Proporcione una recomendación concreta: aprobación, aprobación con condiciones, o rechazo del crédito.
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en análisis de riesgo crediticio para entidades bancarias. "
                    "Tus reportes son profesionales, concisos y basados en evidencia cuantitativa."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model="llama-3.1-8b-instant",
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message.content
