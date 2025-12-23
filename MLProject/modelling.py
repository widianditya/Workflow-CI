import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from mlflow.models import infer_signature

# 1. KONFIGURASI TRACKING URI
if os.getenv('GITHUB_ACTIONS') == 'true':
    
    print("Running in CI: Saving artifacts to GitHub local storage")
else:
    # Lokal: Tetap gunakan localhost untuk Kriteria 2
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Heart_Disease_Final_Project")
mlflow.sklearn.autolog(log_models=False)

# 2. LOAD DATA 
data_path = 'namadataset_preprocessing/cleaned_heart.csv'
df = pd.read_csv(data_path)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)

with mlflow.start_run(run_name="CI_Model_Training"):
    model.fit(X_train, y_train)
    signature = infer_signature(X_test, model.predict(X_test))
    
    # Log model dan register 
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        registered_model_name="HeartDiseaseClassifier",
        signature=signature
    )
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    print(f"✅ Training Success. Accuracy: {acc:.4f}")
