import os
import joblib   
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS ABSOLUTAS (HECHOS)
# =====================================================================
RUTA_TRANSFORMADOR = r"C:\Users\Carlos\Documents\Curso_Analisis_Data_bootcamp_Upgrade_Hub\Inteligencia_Producto_E_Commerce\models\preprocessors\transformador_aduana.pkl"
RUTA_MODELO = r"C:\Users\Carlos\Documents\Curso_Analisis_Data_bootcamp_Upgrade_Hub\Inteligencia_Producto_E_Commerce\models\optimized_models\xgboost_campeon_optimizado.pkl"


# Carga optimizada en memoria
@st.cache_resource
def cargar_componentes():
    if not os.path.exists(RUTA_TRANSFORMADOR) or not os.path.exists(
        RUTA_MODELO
    ):
        st.error(
            "⚠️ Archivos de producción no encontrados. Verifique las rutas locales."
        )
        st.stop()
    with open(RUTA_TRANSFORMADOR, "rb") as f_trans:
        trans = joblib.load(f_trans)
    with open(RUTA_MODELO, "rb") as f_mod:
        mod = joblib.load(f_mod)
    return trans, mod


aduana_datos, xgboost_campeon = cargar_componentes()

# =====================================================================
# 2. INTERFAZ GRÁFICA Y CATÁLOGOS DE DATOS (PRODUCTO BASE)
# =====================================================================
st.set_page_config(page_title="Simulador Predictivo Insurtech", layout="wide")
st.title("🎯 Simulador de Comportamiento de Clientes")
st.write(
    "Introduzca los parámetros del cliente. El sistema validará los datos mediante la aduana matemática antes de la inferencia."
)

# Creamos dos columnas en la interfaz para organizar el formulario
col_cat, col_num = st.columns(2)

# --- COLUMNA 1: VARIABLES CATEGÓRICAS (Catálogos cerrados con búsqueda inteligente) ---
with col_cat:
    st.subheader("📁 Variables Categóricas")

    # Los elementos de estas listas corresponden a los valores únicos reales de tus datos
    country = st.selectbox(
        "1. País de Origen (Soporta escritura para búsqueda):",
        ["BR", "AR", "US", "ES", "CO", "CL", "MX", "PT", "FR", "DE"],
    )

    category = st.selectbox(
        "2. Categoría del Producto:",
        [
            "electronics",
            "fashion",
            "home_appliances",
            "health_beauty",
            "sports",
            "automotive",
        ],
    )

    device_type = st.selectbox(
        "3. Dispositivo de Acceso:", ["mobile", "desktop", "tablet"]
    )

    interaction_type = st.selectbox(
        "4. Tipo de Interacción Principal:",
        ["view", "add_to_wishlist", "click_ad", "purchase_attempt"],
    )

    loyalty_tier = st.selectbox(
        "5. Nivel de Fidelidad (Tier):", ["none", "regular", "silver", "gold"]
    )

# --- COLUMNA 2: VARIABLES NUMÉRICAS (Límites de control de negocio) ---
with col_num:
    st.subheader("🔢 Variables Numéricas")

    age = st.number_input(
        "6. Edad del Cliente:", min_value=10, max_value=99, value=35, step=1
    )

    price = st.number_input(
        "7. Precio del Producto (R$):",
        min_value=0.0,
        max_value=10000.0,
        value=150.0,
        step=10.0,
    )

    rating_avg = st.slider(
        "8. Calificación Promedio del Producto:",
        min_value=1.0,
        max_value=5.0,
        value=4.2,
        step=0.1,
    )

    review_count = st.number_input(
        "9. Volumen de Reseñas del Producto:",
        min_value=0,
        max_value=5000,
        value=120,
        step=5,
    )

    stock_quantity = st.number_input(
        "10. Inventario Disponible (Stock):",
        min_value=0,
        max_value=1000,
        value=45,
        step=1,
    )

    dwell_time_secs = st.slider(
        "11. Tiempo de Permanencia (Segundos en la App):",
        min_value=0,
        max_value=3600,
        value=180,
        step=10,
    )

# =====================================================================
# 3. VECTORIZACIÓN, ADUANA Y PREDICCIÓN
# =====================================================================
st.markdown("---")

if st.button("🚀 Ejecutar Inferencia en Tiempo Real", type="primary"):

    # HECHO: Construcción del DataFrame estructurado exactamente con las 11 columnas ordenadas
    registro_cliente = pd.DataFrame(
        [
            {
                # Numéricas
                "age": age,
                "price": price,
                "rating_avg": rating_avg,
                "review_count": review_count,
                "stock_quantity": stock_quantity,
                "dwell_time_secs": dwell_time_secs,
                # Categóricas
                "country": country,
                "category": category,
                "device_type": device_type,
                "interaction_type": interaction_type,
                "loyalty_tier": loyalty_tier,
            }
        ]
    )

    try:
        # Paso 1: Transformación en la Máquina Moldeadora (Genera las 44 columnas resultantes del OHE/Yeo-Johnson)
        datos_saneados = aduana_datos.transform(registro_cliente)

        # Paso 2: Inferencia en el XGBoost Campeón
        prediccion = xgboost_campeon.predict(datos_saneados)[0]
        probabilidad = xgboost_campeon.predict_proba(datos_saneados)[0][
            1
        ]  # Clase True

        # Paso 3: Despliegue de resultados de gobernanza
        st.subheader("🔬 Diagnóstico de Inferencia")
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                label="Probabilidad de Conversión Estimada",
                value=f"{probabilidad * 100:.2f}%",
            )

        with c2:
            if prediccion == 1:
                st.success(
                    "🔥 ALERTA DE NEGOCIO: Alta probabilidad de conversión (Clase Fiel Detectada)."
                )
            else:
                st.info(
                    "💤 COMPORTAMIENTO PASIVO: El cliente no muestra patrones de propensión inmediata."
                )

    except Exception as e:
        st.error(
            f"Error en la alineación de la matriz en el transformador: {str(e)}"
        )