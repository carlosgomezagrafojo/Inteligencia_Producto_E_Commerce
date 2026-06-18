# 🚀 Inteligencia de Producto E-Commerce: Pipeline MLOps de Propensión de Compra

Este repositorio contiene una solución de producción de extremo a extremo (**End-to-End**) para predecir la propensión de conversión de clientes en una plataforma de e-commerce en tiempo real. El proyecto destaca por su enfoque en **MLOps**, implementando un flujo circular vivo que captura logs de inferencia, evalúa umbrales de masa crítica y dispara re-entrenamientos automatizados con versionado seguro de modelos.

Desplegado en producción web a través de **Streamlit Cloud**.

---

## 📊 Arquitectura del Sistema (El Bucle Circular)

A diferencia de los modelos analíticos estáticos que mueren en un cuaderno de Jupyter, esta infraestructura implementa un **Ciclo de Vida Continuo de la IA (Continuous Training - CT)** desacoplado en tres capas operativas:

1. **Capa de Inferencia y Captura (App Web):** El usuario introduce los parámetros del cliente. La app valida los datos mediante una aduana matemática (`ColumnTransformer`), predice el score estratégico mediante un modelo **XGBoost** optimizado y escribe un log inmediato en disco de forma ultra-eficiente.
2. **Capa de Ingesta y Orquestación (`trigger_reentrenamiento.py`):** Un guardián asíncrono monitorea el volumen de registros utilizando operaciones de lectura a nivel de sistema operativo (I/O de bajo consumo). Al alcanzar la masa crítica estipulada (ej. cada 50 registros), empaqueta y sanitiza la materia prima en formato comprimido **Parquet**.
3. **Capa de Re-entrenamiento e Historial (`reentrenamiento_automatico.py`):** El motor absorbe los nuevos datos, los fusiona con el histórico original (`X_train` / `y_train`), ajusta el algoritmo con sus hiperparámetros óptimos y ejecuta un **reemplazo en caliente** del modelo de producción, respaldando automáticamente el cerebro anterior en una carpeta de archivo (`models/archive/`) con su marca de tiempo exacta.

---

## 🛠️ Stack Tecnológico Utilizado

* **Lenguaje:** Python 3.11+
* **Modelado Predictivo:** XGBoost, Scikit-Learn (ColumnTransformer, OneHotEncoder, PowerTransformer Yeo-Johnson).
* **Ingeniería de Datos:** Pandas, Numpy, Joblib, PyArrow (Formatos Parquet eficientes).
* **Interfaz de Producción:** Streamlit Cloud (Despliegue asíncrono con `subprocess`).
* **Control de Versiones y MLOps:** Git, GitHub, Model Registry local/archive.

---

## 📁 Estructura del Proyecto

```text
Inteligencia_Producto_E_Commerce/
│
├── .github/                      # Configuraciones de despliegue en la nube
├── data/
│   ├── processed/                # Datasets históricos (X_train, y_train) y logs en producción
│   └── raw/                      # Dataset original del negocio
│
├── models/
│   ├── archive/                  # 🛡️ Historial y versionado de modelos anteriores (Copia de seguridad)
│   ├── optimized_models/         # Cerebro activo en producción (xgboost_campeon_optimizado.pkl)
│   └── preprocessors/            # Aduana matemática (transformador_aduana.pkl)
│
├── app.py                        # Interfaz de usuario y consola de ingeniería en Streamlit
├── trigger_reentrenamiento.py    # Orquestador asíncrono en segundo plano
├── reentrenamiento_automatico.py # Motor de re-entrenamiento circular con inyección de datos
├── requirements.txt              # Dependencias del entorno de producción congeladas
└── README.md                     # Documentación técnica del proyecto

