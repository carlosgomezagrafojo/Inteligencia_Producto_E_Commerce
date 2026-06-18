import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_MATERIA_PRIMA = os.path.join(BASE_DIR, "data", "processed", "materia_prima_reentrenamiento.parquet")
RUTA_HISTORICO_X = os.path.join(BASE_DIR, "data", "processed", "X_train_saneado.parquet")
RUTA_HISTORICO_Y = os.path.join(BASE_DIR, "data", "processed", "y_train.parquet")
RUTA_TRANSFORMADOR = os.path.join(BASE_DIR, "models", "preprocessors", "transformador_aduana.pkl")
RUTA_MODELO_CAMPEON = os.path.join(BASE_DIR, "models", "optimized_models", "xgboost_campeon_optimizado.pkl")
CARPETA_RESPALDOS = os.path.join(BASE_DIR, "models", "archive")



# Nueva carpeta para que no se pierda ningún modelo del pasado
CARPETA_RESPALDOS = r"C:\Users\Carlos\Documents\Curso_Analisis_Data_bootcamp_Upgrade_Hub\Inteligencia_Producto_E_Commerce\models\archive"

def ejecutar_bucle_reentrenamiento():
    print("=" * 80)
    print("🧠 INICIANDO BUCLE DE RE-ENTRENAMIENTO CIRCULAR CON VERSIONADO")
    print("=" * 80)
    
    if not os.path.exists(RUTA_MATERIA_PRIMA):
        print("🛑 ABORTANDO: No hay materia prima nueva para procesar.")
        return False
        
    # Crear la carpeta de archivos históricos si no existe
    os.makedirs(CARPETA_RESPALDOS, exist_ok=True)
    
    # 2. Carga de componentes de producción
    print("⚙️ Cargando aduana matemática y modelo campeón actual...")
    aduana = joblib.load(RUTA_TRANSFORMADOR)
    modelo_actual = joblib.load(RUTA_MODELO_CAMPEON)
    
    # 3. Fusión de Datos (Expandiendo la experiencia del modelo)
    print("📚 Uniendo histórico de entrenamiento con los datos nuevos del Streamlit...")
    df_nuevos = pd.read_parquet(RUTA_MATERIA_PRIMA)
    X_historico = pd.read_parquet(RUTA_HISTORICO_X)
    y_historico = pd.read_parquet(RUTA_HISTORICO_Y)
    
    # Concatenamos los datos nuevos debajo de los históricos
    X_consolidado = pd.concat([X_historico, df_nuevos], ignore_index=True)
    
    # Simulamos las etiquetas para mantener la integridad del laboratorio
    y_nuevas = pd.DataFrame({"is_converted": np.random.choice([0, 1], size=len(df_nuevos), p=[0.8, 0.2])})
    y_consolidado = pd.concat([y_historico, y_nuevas], ignore_index=True)
    
    print(f"📊 Dataset expandido con éxito de {X_historico.shape[0]} a {X_consolidado.shape[0]} filas.")
    
    # 4. Transformación de Aduana
    X_trans = aduana.transform(X_consolidado)
    
    # 5. Clonación de hiperparámetros y Entrenamiento del Nuevo Cerebro
    print("🏋️‍♂️ Entrenando el nuevo XGBoost con el universo expandido...")
    hiperparametros = modelo_actual.get_params()
    nuevo_modelo = XGBClassifier(**hiperparametros)
    nuevo_modelo.fit(X_trans, y_consolidado.values.ravel())
    
    # 6. 🛡️ MEDIDA DE SEGURIDAD CARLOS: Versionado e Historial del Modelo Viejo
    timestamp_modelo = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    RUTA_RESPALDO_MODELO = os.path.join(CARPETA_RESPALDOS, f"xgboost_modelo_version_{timestamp_modelo}.pkl")
    
    print(f"📦 Respaldando el modelo anterior en el historial: {RUTA_RESPALDO_MODELO}")
    joblib.dump(modelo_actual, RUTA_RESPALDO_MODELO) # Guardamos el viejo en el archivo
    
    # 7. Reemplazo Seguro para Producción
    print("💾 Desplegando el nuevo cerebro en producción (Reemplazo en Caliente)...")
    joblib.dump(nuevo_modelo, RUTA_MODELO_CAMPEON) # El nuevo toma el control
    
    # 8. Limpieza
    os.remove(RUTA_MATERIA_PRIMA)
    print("Sweep 🧹: Archivo temporal limpiado. Sistema listo para el próximo ciclo.")
    print("=" * 80)
    print("🎉 [ÉXITO]: Bucle cerrado. Tienes copia del modelo viejo y la app ya usa el nuevo.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    ejecutar_bucle_reentrenamiento()