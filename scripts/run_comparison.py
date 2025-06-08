# scripts/run_comparison.py

import os
import pandas as pd

# Rutas
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ae_path = os.path.join(project_root, "results", "autoencoder", "autoencoder_anomaly_results.csv")
if_path = os.path.join(project_root, "results", "isolation_forest", "pca_isolation_forest_results.csv")
out_path = os.path.join(project_root, "results", "comparison")
os.makedirs(out_path, exist_ok=True)

# Cargar resultados
df_auto = pd.read_csv(ae_path)
df_iforest = pd.read_csv(if_path)

# Validación
assert len(df_auto) == len(df_iforest), "❌ Número de estrellas no coincide entre autoencoder y Isolation Forest"

# Unir
df_auto['Star'] = df_iforest['Star']
df_auto.rename(columns={'Anomalia': 'Anom_Autoencoder'}, inplace=True)
df_iforest.rename(columns={'Anomaly': 'Anom_IsolationForest'}, inplace=True)

df_comparacion = pd.merge(
    df_auto[['Star', 'Anom_Autoencoder']],
    df_iforest[['Star', 'Anom_IsolationForest']],
    on='Star'
)

# Mostrar y guardar
print("📊 Comparación de métodos:")
print(df_comparacion)

df_comparacion.to_csv(os.path.join(out_path, "comparacion_modelos.csv"), index=False)
print("✅ Comparación guardada en 'results/comparison/comparacion_modelos.csv'")
