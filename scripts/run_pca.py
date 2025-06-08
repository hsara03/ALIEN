# scripts/run_pca.py

import os
from src.pca_analysis import (
    load_processed_data,
    standardize_data,
    apply_pca,
    plot_pca_results,
    save_pca_results
)

# 🧭 Rutas absolutas al proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_input = os.path.join(project_root, "data", "processed", "train")
test_input = os.path.join(project_root, "data", "processed", "test")

train_output = os.path.join(project_root, "results", "pca", "train")
test_output = os.path.join(project_root, "results", "pca", "test")
os.makedirs(train_output, exist_ok=True)
os.makedirs(test_output, exist_ok=True)

# ✅ Análisis PCA para entrenamiento
print("🔬 PCA sobre curvas de entrenamiento...")
train_matrix = load_processed_data(train_input)
train_matrix_std = standardize_data(train_matrix)
train_pcs, train_var = apply_pca(train_matrix_std)

save_pca_results(train_pcs, os.path.join(train_output, "pca_results.csv"))
plot_pca_results(train_pcs, train_var)

# ✅ Análisis PCA para test
print("🔬 PCA sobre curvas de test...")
test_matrix = load_processed_data(test_input)
test_matrix_std = standardize_data(test_matrix)
test_pcs, test_var = apply_pca(test_matrix_std)

save_pca_results(test_pcs, os.path.join(test_output, "pca_results.csv"))
plot_pca_results(test_pcs, test_var)

print("✅ Análisis PCA finalizado.")
