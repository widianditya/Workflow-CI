import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from mlflow.models import infer_signature

# ==========================================
# 1. KONFIGURASI TRACKING URI (Krusial!)
# ==========================================
# Kode ini mendeteksi apakah berjalan di GitHub Actions atau Laptop Anda
if os.getenv('GITHUB_ACTIONS') == 'true':
    # LINGKUNGAN GITHUB: Kirim data ke DagsHub (Kriteria 3)
    dagshub.init(
        repo_owner='widianditya', 
        repo_name='Eksperimen_SML_I-Gede-Made-Widi-Anditya', 
        mlflow=True
    )
    print("Running on GitHub Actions: Tracking to DagsHub")
else:
    # LINGKUNGAN LOKAL: Kirim data ke Localhost (Kriteria 2)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print("Running Locally: Tracking to http://127.0.0.1:5000")

# Set nama eksperimen agar rapi di dashboard
mlflow.set_experiment("Heart_Disease_Final_Project")

# 2. Konfigurasi Autologging (Hanya log parameter, model kita log manual)
mlflow.sklearn.autolog(log_models=False)

# ==========================================
# 3. PREPARASI DATA
# ==========================================
data_path = 'namadataset_preprocessing/cleaned_heart.csv'
df = pd.read_csv(data_path)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. TRAINING & LOGGING
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7, 
    criterion='gini', 
    random_state=42
)

with mlflow.start_run(run_name="Final_Model_Submission"):
    model.fit(X_train, y_train)
    
    # Membuat signature (skema input/output)
    signature = infer_signature(X_test, model.predict(X_test))
    
    # --- BAGIAN REVISI REVIEWER (KRITERIA 2) ---
    # Log model secara eksplisit ke folder "model" dan DAFTARKAN modelnya
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        registered_model_name="HeartDiseaseClassifier",
        signature=signature
    )
    
    # --- LOGGING METRIK & ARTEFAK TAMBAHAN ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # Simpan Confusion Matrix sebagai gambar artefak
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['Healthy', 'Sick'], ax=ax)
    plt.title("Confusion Matrix - Test Data")
    
    plot_path = "training_confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
    print(f"✅ Success! Model Registered. Accuracy: {acc:.4f}")
