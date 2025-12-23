import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

# 1. Konfigurasi Tracking URI (GitHub Actions vs Lokal)
if os.getenv('GITHUB_ACTIONS') == 'true':
    # Lingkungan CI/CD: Kirim data ke DagsHub
    dagshub.init(repo_owner='widianditya', 
                 repo_name='Eksperimen_SML_I-Gede-Made-Widi-Anditya', 
                 mlflow=True)
else:
    # Lingkungan Lokal: Gunakan localhost sesuai syarat Kriteria 2
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Konfigurasi Autologging
mlflow.sklearn.autolog(log_models=False)

# 3. Load Dataset
data_path = 'namadataset_preprocessing/cleaned_heart.csv'
df = pd.read_csv(data_path)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi Model dengan Parameter Terbaik
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7, 
    criterion='gini', 
    random_state=42
)

# 5. Eksekusi Training dan Logging
with mlflow.start_run(run_name="HeartDisease_Final_Attempt"):
    model.fit(X_train, y_train)
    
    # 1: Log Model dengan Nama Folder Spesifik
    # Ini akan membuat folder 'best_heart_model' di DagsHub Artifacts
    mlflow.sklearn.log_model(sk_model=model, artifact_path="best_heart_model")
    
    # 2: Confusion Matrix dari Data UJI
    # Autolog membuat versi data training, ini adalah versi data uji (test data)
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['Healthy', 'Sick'], ax=ax)
    plt.title("Confusion Matrix - Test Data")
    
    # Simpan dan unggah sebagai artefak
    plot_path = "test_confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
    # 3: Logging Metrik untuk Dashboard
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    print(f"Final Model logged. Accuracy: {acc:.4f}, F1: {f1:.4f}")