import mlflow
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk mencetak pesan dengan pemisah agar mudah dibaca di log
def print_header(message):
    print("\n" + "="*50)
    print(f" {message}")
    print("="*50)

if __name__ == "__main__":
    # Setup parser untuk menerima argumen dari command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed dataset.")
    args = parser.parse_args()

    print_header("Memulai Training Model untuk CI/CD")

    # MLflow akan secara otomatis menangani tracking URI di GitHub Actions
    # jadi kita tidak perlu mengaturnya secara manual di sini.
    # MLflow run akan membuat folder mlruns di root direktori.
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Memulai MLflow run
    with mlflow.start_run():
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        
        # 1. Muat Dataset
        print_header("1. Memuat Dataset")
        try:
            data = pd.read_csv(args.data_path)
            # Pastikan tidak ada spasi tersembunyi pada nama kolom
            data.columns = data.columns.str.strip()
            print(f"Dataset berhasil dimuat dari: {args.data_path}")
            print(f"Bentuk data: {data.shape}")
            print(f"Nama kolom: {list(data.columns)[:10]} ... total {len(data.columns)} kolom")
        except FileNotFoundError:
            print(f"Error: Dataset tidak ditemukan di path: {args.data_path}")
            exit(1)

        # 2. Tentukan kolom target secara robust
        print_header("2. Menentukan Kolom Target")
        candidate_targets = ["type_encoded", "Class", "class", "Label", "label", "target", "Target", "y"]
        target_col = next((c for c in candidate_targets if c in data.columns), None)
        
        if target_col is None:
            # Coba cari kolom biner (0/1) sebagai fallback
            binary_candidates = []
            for c in data.columns:
                try:
                    unique_vals = set(pd.Series(data[c]).dropna().unique())
                    if unique_vals.issubset({0, 1}) and 1 in unique_vals:
                        binary_candidates.append(c)
                except Exception:
                    pass
            if binary_candidates:
                target_col = binary_candidates[0]
                print(f"Target tidak ditemukan secara eksplisit. Menggunakan kolom biner: {target_col}")
            else:
                print("Error: Kolom target tidak ditemukan. Pastikan ada salah satu dari ['type_encoded','Class','Label','target','y'] atau kolom biner (0/1).")
                print(f"Daftar kolom saat ini: {list(data.columns)}")
                exit(1)
        else:
            print(f"Menggunakan kolom target: {target_col}")
        
        # 3. Split fitur dan target
        X = data.drop(columns=[target_col])
        y = data[target_col]

        print_header("3. Memisahkan Data Training dan Testing")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Data training: {X_train.shape[0]} sampel")
        print(f"Data testing: {X_test.shape[0]} sampel")
        
        # 4. Training Model
        print_header("4. Melatih Model RandomForestClassifier")
        # Definisikan parameter
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        # Log parameter ke MLflow
        mlflow.log_params(params)
        print("Parameter yang digunakan:", params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        print("Model selesai dilatih.")
        
        # 5. Evaluasi Model
        print_header("5. Mengevaluasi Model")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {accuracy:.4f}")

        # Log metrik ke MLflow
        mlflow.log_metric("accuracy", accuracy)
        
        # 6. Log Model
        print("Menyimpan model ke MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="CreditScoringCICD" # Nama model untuk Model Registry
        )
        print("Model berhasil disimpan dengan nama 'CreditScoringCICD'.")

    print_header("Proses Training Selesai")