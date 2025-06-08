# src/preprocessings.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_lightcurve(file_path):
    """
    Carga una curva de luz desde CSV y devuelve un DataFrame con columnas 'tiempo' y 'brillo'.
    Usa la mejor columna de tiempo disponible ('timecorr' > 'cadenceno' > 'time').
    Normaliza la columna de brillo si es necesario.
    """
    df = pd.read_csv(file_path)

    # Filtrar calidad si existe
    if 'quality' in df.columns:
        df = df[df['quality'] == 0].reset_index(drop=True)

    # Determinar columna de tiempo
    if 'timecorr' in df.columns:
        time_col = 'timecorr'
    elif 'cadenceno' in df.columns:
        df['cadenceno'] -= df['cadenceno'].min()
        time_col = 'cadenceno'
    elif 'time' in df.columns:
        time_col = 'time'
    else:
        raise KeyError(f"No se encontró ninguna columna de tiempo válida en {file_path}.")

    # Determinar columna de brillo
    if 'flux' in df.columns:
        df['brillo'] = df['flux']
    elif 'pdcsap_flux' in df.columns:
        df['brillo'] = df['pdcsap_flux'] / np.median(df['pdcsap_flux'])
    else:
        raise KeyError(f"No se encontró ninguna columna de brillo válida en {file_path}.")

    # Interpolar si hay NaNs
    df['brillo'] = df['brillo'].interpolate(method='linear').fillna(df['brillo'].median())

    return df[[time_col, 'brillo']].rename(columns={time_col: 'tiempo'})


def plot_lightcurve(df, title="Curva de Luz", ylim=None):
    """Grafica una curva de luz dada, con límites opcionales."""
    plt.figure(figsize=(10, 5))
    plt.plot(df['tiempo'], df['brillo'], 'k.', markersize=1, alpha=0.5)
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Brillo Normalizado')
    plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_removed_outliers(original_df, filtered_df):
    """Grafica los puntos eliminados tras quitar outliers."""


    merged = original_df.merge(filtered_df, on=['tiempo', 'brillo'], how='left', indicator=True)
    eliminados = merged[merged['_merge'] == 'left_only']

    plt.figure(figsize=(10, 5))
    plt.plot(original_df['tiempo'], original_df['brillo'], 'k.', markersize=1, alpha=0.5, label='Original')
    plt.plot(eliminados['tiempo'], eliminados['brillo'], 'ro', markersize=2, alpha=0.8, label='Outliers eliminados')
    plt.xlabel('Tiempo (días)')
    plt.ylabel('Brillo Normalizado')
    plt.title('Outliers eliminados sobre la curva original')
    plt.legend()
    plt.tight_layout()
    plt.show()


def remove_isolated_outliers(df, window=10, sigma=2):
    """
    Elimina puntos atípicos aislados comparando brillo con media local en una ventana.
    """
    df_filtered = df.copy()
    df_filtered['rolling_mean'] = df_filtered['brillo'].rolling(window=window, center=True).mean()
    df_filtered['rolling_std'] = df_filtered['brillo'].rolling(window=window, center=True).std()

    limite_sup = df_filtered['rolling_mean'] + sigma * df_filtered['rolling_std']
    limite_inf = df_filtered['rolling_mean'] - sigma * df_filtered['rolling_std']

    outliers = (df_filtered['brillo'] > limite_sup) | (df_filtered['brillo'] < limite_inf)
    df_filtered = df_filtered[~outliers]

    print(f"🧹 Se eliminaron {len(df) - len(df_filtered)} outliers.")

    return df_filtered.drop(columns=['rolling_mean', 'rolling_std'])


def save_processed_data(df, output_path):
    """Guarda una curva procesada como CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Guardado en: {output_path}")
