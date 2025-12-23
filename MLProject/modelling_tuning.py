import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import shutil

# Ambil token dari environment variable jika tersedia
token = os.getenv("DAGSHUB_USER_TOKEN")

if token:
    # Jika berjalan di GitHub Actions, gunakan token untuk login otomatis
    os.environ['MLFLOW_TRACKING_USERNAME'] = token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    print("Berjalan di CI: Menggunakan DAGSHUB_USER_TOKEN untuk otentikasi.")

# 1. Inisialisasi DagsHub
REPO_OWNER = 'widianditya'
REPO_NAME = 'Eksperimen_SML_I-Gede-Made-Widi-Anditya'
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

# 2. Load Data
df = pd.read_csv('namadataset_preprocessing/cleaned_heart.csv')
df.columns = df.columns.str.strip().str.lower()

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Hyperparameter Tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7], 
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("Sedang mencari kombinasi parameter terbaik... Mohon tunggu.")
grid_search.fit(X_train, y_train)

# 4. Mengambil Model Terbaik
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 5. Logging ke MLflow (Syarat Advance)
with mlflow.start_run(run_name="Optimized_RandomForest_90Percent"):
    
    # Log Parameter Terbaik hasil Tuning
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("tuning_method", "GridSearchCV")

    # Log Metrik Akhir
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log Artefak: Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
    plt.title(f'Optimized Model (Acc: {acc:.2f})')
    plt.savefig("optimized_confusion_matrix.png")
    mlflow.log_artifact("optimized_confusion_matrix.png")

    # Log Model
    mlflow.sklearn.log_model(best_model, "best_heart_model")
    
    # 1. Tentukan lokasi folder model di dalam direktori lokal
    model_path_lokal = "Membangun_model/best_heart_model"

    # 2. Hapus folder jika sudah ada (mencegah error 'Directory already exists')
    if os.path.exists(model_path_lokal):
        shutil.rmtree(model_path_lokal)

    # 3. Simpan model secara fisik ke folder lokal
    mlflow.sklearn.save_model(sk_model=best_model, path=model_path_lokal)
    
    print(f"Model fisik berhasil disimpan di: {model_path_lokal}")

    print(f"Selesai! Akurasi terbaik yang didapat: {acc:.4f}")
    print(f"Parameter terbaik: {grid_search.best_params_}")