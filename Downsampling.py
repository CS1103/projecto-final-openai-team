import pandas as pd
from sklearn.utils import resample

# 1. Leer el dataset original
df = pd.read_csv("diabetes_prediction_dataset.csv")

# 2. Separar clases
majority = df[df['diabetes'] == 0]
minority = df[df['diabetes'] == 1]

# 3. Submuestrear la clase mayoritaria a 8500
majority_downsampled = resample(majority,
                                replace=False,
                                n_samples=8500,
                                random_state=42)

# 4. Combinar
df_balanced = pd.concat([majority_downsampled, minority])

# 5. Mezclar aleatoriamente
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Verificar resultado
print(df_balanced['diabetes'].value_counts())

# 7. Guardar a CSV
df_balanced.to_csv("diabetes_prediction_dataset_balanced_8500.csv", index=False)

print("✅ Dataset balanceado guardado como diabetes_prediction_dataset_balanced_8500.csv")
