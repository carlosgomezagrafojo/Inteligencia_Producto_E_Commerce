import os
import pandas as pd


# CONFIGURACIÓN DE RUTAS ABSOLUTAS 

RUTA_LOGS = r"C:\Users\Carlos\Documents\Curso_Analisis_Data_bootcamp_Upgrade_Hub\Inteligencia_Producto_E_Commerce\data\processed\logs_inferencia.csv"
RUTA_ENTRENAMIENTO_HISTORICO = r"C:\Users\Carlos\Documents\Curso_Analisis_Data_bootcamp_Upgrade_Hub\Inteligencia_Producto_E_Commerce\data\processed\X_train_saneado.parquet"
RUTA_DESTINO_REENTRENAMIENTO = r"C:\Users\Carlos\Documents\Curso_Analisis_Data_bootcamp_Upgrade_Hub\Inteligencia_Producto_E_Commerce\data\processed\materia_prima_reentrenamiento.parquet"

def ejecutar_orquestacion_datos():
    print("=" * 80)
    print("🚀 EJECUTANDO ORQUESTADOR DE DATOS PARA RE-ENTRENAMIENTO")
    print("=" * 80)
    
    # 1. Validación de existencia de activos
    if not os.path.exists(RUTA_LOGS):
        print("❌ ERROR: No se encontró el archivo de logs históricos.")
        return False
        
    # 2. Carga eficiente de la materia prima nueva (Logs acumulados en la App)
    print("📋 Leyendo logs acumulados desde la aplicación...")
    df_logs_nuevos = pd.read_csv(RUTA_LOGS)
    
    # ⚙️ REGLA DE NEGOCIO: Separamos las características de la predicción de la App
    # Eliminamos las columnas de control que metió la app ('pred_probabilidad' y 'timestamp')
    # para dejar el dataset exactamente igual a como el transformador original lo necesita.
    columnas_a_remover = ["pred_probabilidad", "timestamp"]
    df_nuevos_limpios = df_logs_nuevos.drop(columns=columnas_a_remover, errors="ignore")
    
    print(f"🔹 Registros nuevos listos para ser procesados: {len(df_nuevos_limpios)}")
    
    # 3. Almacenamiento aislado de seguridad
    # Guardamos este bloque en un archivo Parquet intermedio para que el Cuaderno 3 pueda absorberlo
    df_nuevos_limpios.to_parquet(RUTA_DESTINO_REENTRENAMIENTO, index=False)
    print(f"💾 Materia prima exportada con éxito en: {RUTA_DESTINO_REENTRENAMIENTO}")
    
    # LLAMADA AL BUCLE DE RE-ENTRENAMIENTO
    print("🔌 Conectando con el motor de re-entrenamiento...")
    import subprocess
    import sys
    
    try:
        # El orquestador ejecuta el script del nuevo cerebro de forma secuencial
        subprocess.run([sys.executable, "reentrenamiento_automatico.py"], check=True)
        print("🟢 [HECHO]: El bucle circular se ejecutó y cerró con éxito.")
    except Exception as e:
        print(f"❌ ERROR al ejecutar el re-entrenamiento: {str(e)}")
        return False

    print("=" * 80)
    return True

if __name__ == "__main__":
    ejecutar_orquestacion_datos()