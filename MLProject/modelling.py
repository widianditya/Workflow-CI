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

# 1. Konfigurasi Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart_Disease_Final_Project")

# 2. Konfigurasi Autologging
mlflow.sklearn.autolog(log_models=False)

# 3. Load Dataset
data_path = 'namadataset_preprocessing/cleaned_heart.csv'
df = pd.read_csv(data_path)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7, 
    criterion='gini', 
    random_state=42
)

# 5. Eksekusi Training dan Logging
with mlflow.start_run(run_name="Registered_Model_Attempt"):
    model.fit(X_train, y_train)
    
    # 1: Menentukan Signature (Skema Input/Output) untuk Artefak
    signature = infer_signature(X_test, model.predict(X_test))
    
    # 2: LOG MODEL SECARA EKSPLISIT 
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        registered_model_name="HeartDiseaseClassifier",
        signature=signature
    )
    
    # 3: Log Confusion Matrix sebagai Artefak Tambahan
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['Healthy', 'Sick'], ax=ax)
    plt.title("Confusion Matrix - Test Data")
    
    plot_path = "training_confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
    # 4: Logging Metrik
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    print(f"Model Registered Successfully. Accuracy: {acc:.4f}")
