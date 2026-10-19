import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan di: {input_path}")

    # 1. Memuat Dataset
    df = pd.read_csv(input_path)
    df_prep = df.copy()

    # 2. Menghapus Data Duplikat
    df_prep = df_prep.drop_duplicates()

    # 3. Menangani Missing Values (Imputasi nilai 0 dengan Median)
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        df_prep[col] = df_prep[col].replace(0, np.nan)
        df_prep[col] = df_prep[col].fillna(df_prep[col].median())

    # 4. Penanganan Outlier (IQR Clipping)
    feature_cols = [c for c in df_prep.columns if c != 'Outcome']
    for col in feature_cols:
        Q1 = df_prep[col].quantile(0.25)
        Q3 = df_prep[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_prep[col] = df_prep[col].clip(lower=lower_bound, upper=upper_bound)

    # 5. Standarisasi Fitur (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_prep[feature_cols])

    df_clean = pd.DataFrame(X_scaled, columns=feature_cols)
    df_clean['Outcome'] = df_prep['Outcome'].values

    # 6. Menyimpan Hasil Preprocessing
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Preprocessing otomatis selesai! Tersimpan di: {output_path} (Ukuran: {df_clean.shape})")

    return df_clean

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "..", "diabetes_raw.csv")
    out_path = os.path.join(base_dir, "diabetes_preprocessing.csv")
    preprocess_data(raw_path, out_path)