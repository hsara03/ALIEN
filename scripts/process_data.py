# scripts/process_data.py

import os
from src.preprocessings import (
    load_lightcurve,
    remove_isolated_outliers,
    save_processed_data,
    plot_lightcurve,
    plot_removed_outliers
)

# 🧭 Rutas absolutas seguras
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, "data", "raw")
train_path = os.path.join(project_root, "data", "processed", "train")
test_path = os.path.join(project_root, "data", "processed", "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 📄 Archivos a separar manualmente en train y test
train_stars = [
    "Kepler-10", "Kepler-22", "Kepler-90", "KIC_8462852"
]
test_stars = [
    "Kepler-62", "KIC_8197761", "KIC_12557548", "KIC_3544595"
]

# 🔄 Recorremos todos los archivos
for filename in os.listdir(raw_data_path):
    if not filename.endswith(".csv"):
        continue

    full_path = os.path.join(raw_data_path, filename)

    # Extraer el nombre base sin 'curva_luz_' ni '.csv'
    star_name = filename.replace("curva_luz_", "").replace(".csv", "")

    print(f"\n🔍 Procesando: {star_name}")

    try:
        df = load_lightcurve(full_path)
        df_filtered = remove_isolated_outliers(df)

        # Opcional: mostrar gráfico de limpieza
        plot_removed_outliers(df, df_filtered)

        # Determinar destino
        if star_name in train_stars:
            output_path = os.path.join(train_path, filename)
        elif star_name in test_stars:
            output_path = os.path.join(test_path, filename)
        else:
            print(f"⚠️ {star_name} no está en ninguna lista. Saltando.")
            continue

        save_processed_data(df_filtered, output_path)

    except Exception as e:
        print(f"❌ Error procesando {filename}: {e}")
