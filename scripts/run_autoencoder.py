# scripts/run_autoencoder.py

import os
import numpy as np
import pandas as pd
from src.pca_analysis import load_processed_data, standardize_data
from src.autoencoder_model import (
    build_autoencoder,
    train_autoencoder,
    compute_reconstruction_errors,
    plot_reconstruction_error,
    detect_anomalies
)

# Rutas
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(project_root, "data", "processed", "train")
result_dir = os.path.join(project_root, "results", "autoencoder")
os.makedirs(result_dir, exist_ok=True)

# 1️⃣ Cargar y normalizar datos
print("📥 Cargando curvas de entrenamiento...")
data_matrix = load_processed_data(train_dir)
standardized_data = standardize_data(data_matrix)

# 2️⃣ Crear y entrenar autoencoder
print("🧠 Entrenando Autoencoder...")
autoencoder = build_autoencoder(input_dim=standardized_data.shape[1])
train_autoencoder(autoencoder, standardized_data, epochs=50, batch_size=8)

# 3️⃣ Obtener errores de reconstrucción
print("📈 Calculando errores de reconstrucción...")
errors = compute_reconstruction_errors(autoencoder, standardized_data)
plot_reconstruction_error(errors)

# 4️⃣ Detectar anomalías (top 10% por error)
threshold = np.percentile(errors, 90)
labels = detect_anomalies(errors, threshold)

# 5️⃣ Guardar resultados
df = pd.DataFrame({'ErrorReconstruccion': errors, 'Anomalia': labels})
df.to_csv(os.path.join(result_dir, 'autoencoder_anomaly_results.csv'), index=False)
print("✅ Resultados guardados en 'results/autoencoder'")
