# src/pca_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_processed_data(data_dir, num_points=300):
    """
    Carga todas las curvas procesadas desde `data_dir`, las interpola
    a una longitud fija (por defecto 300 puntos) y devuelve una matriz NumPy.
    """
    all_curves = []

    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)

        if 'brillo' not in df.columns:
            print(f"⚠️ {filename} no tiene columna 'brillo'. Saltando...")
            continue

        if df.shape[0] < 2:
            print(f"⚠️ {filename} tiene muy pocos datos. Saltando...")
            continue

        original_x = np.linspace(0, 1, num=len(df))
        target_x = np.linspace(0, 1, num=num_points)
        interpolated_flux = np.interp(target_x, original_x, df['brillo'].values)

        all_curves.append(interpolated_flux)

    return np.array(all_curves)


def standardize_data(data_matrix):
    """
    Estandariza los datos para que cada componente tenga media 0 y varianza 1.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data_matrix)


def apply_pca(data_matrix, n_components=2):
    """
    Aplica PCA a la matriz dada y devuelve las componentes principales
    y la varianza explicada.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_matrix)
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance


def plot_pca_results(principal_components, explained_variance):
    """
    Grafica los datos proyectados en el espacio de las dos primeras componentes principales.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7)
    plt.title(f'PCA de curvas de luz\nVarianza explicada: PC1 {explained_variance[0]:.2f}, PC2 {explained_variance[1]:.2f}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_pca_results(principal_components, output_path):
    """
    Guarda las componentes principales como CSV en la ruta indicada.
    """
    df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Resultados de PCA guardados en: {output_path}")
