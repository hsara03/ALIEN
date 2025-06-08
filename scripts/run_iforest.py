# scripts/run_iforest.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Rutas base
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_pca_path = os.path.join(project_root, "results", "pca", "train", "pca_results.csv")
test_pca_path = os.path.join(project_root, "results", "pca", "test", "pca_results.csv")

train_dir = os.path.join(project_root, "data", "processed", "train")
test_dir = os.path.join(project_root, "data", "processed", "test")

result_dir = os.path.join(project_root, "results", "isolation_forest")
os.makedirs(result_dir, exist_ok=True)

# 1️⃣ Cargar datos PCA
df_train = pd.read_csv(train_pca_path)
df_test = pd.read_csv(test_pca_path)

# 2️⃣ Asignar nombres de estrellas a partir de los archivos
train_names = sorted([f.replace("curva_luz_", "").replace(".csv", "") for f in os.listdir(train_dir) if f.endswith(".csv")])
test_names = sorted([f.replace("curva_luz_", "").replace(".csv", "") for f in os.listdir(test_dir) if f.endswith(".csv")])

df_train['Star'] = train_names[:len(df_train)]
df_test['Star'] = test_names[:len(df_test)]

# 3️⃣ Entrenar modelo
print("🚀 Entrenando Isolation Forest...")
iso_forest = IsolationForest(contamination='auto', random_state=42)
iso_forest.fit(df_train[['PC1', 'PC2']])

# 4️⃣ Aplicar sobre test
print("🔍 Detectando anomalías en test...")
df_test['Anomaly'] = iso_forest.predict(df_test[['PC1', 'PC2']])
df_test['Score'] = iso_forest.decision_function(df_test[['PC1', 'PC2']])

# 5️⃣ Visualización interactiva
fig = px.scatter(
    df_test,
    x='PC1',
    y='PC2',
    color='Anomaly',
    hover_data=['Star', 'Score'],
    title='🔍 Detección de Anomalías con Isolation Forest',
    labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2', 'Anomaly': 'Anomalía'},
    template='plotly_white'
)
fig.update_traces(marker=dict(size=12))
fig.show()

# 6️⃣ Guardar resultados
df_test.to_csv(os.path.join(result_dir, "pca_isolation_forest_results.csv"), index=False)
fig.write_html(os.path.join(result_dir, "isolation_forest_plot.html"))

print("✅ Resultados guardados en carpeta 'results/isolation_forest'")
