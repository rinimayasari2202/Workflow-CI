import mlflow
import dagshub

# Inisialisasi DagsHub di baris paling atas
dagshub.init(repo_owner='rinimayasari2202', repo_name='Workflow-CI', mlflow=True)

import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Set the MLflow Tracking URI to your local server
#mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# 2. Create an MLflow Experiment
#mlflow.set_experiment("Student_Grade_Prediction")

# 3. Enable Autologging
mlflow.autolog()

# 4. Load the preprocessed dataset
print("Loading data...")
try:
    data = pd.read_csv("student_cleaned_automated.csv")
except FileNotFoundError:
    print("ERROR: File CSV tidak ditemukan di folder ini! Pastikan file sudah di-copy.")
    exit()

print("\nKolom yang terbaca oleh sistem:", data.columns.tolist()[:10], "...(dan lainnya)")

# 5. FOOLPROOF FIX: Jika G3_category hilang tapi G3 ada, kita buat ulang di sini
if 'G3_category' not in data.columns:
    if 'G3' in data.columns:
        print("\nTarget 'G3_category' hilang! Membuat ulang target dari kolom 'G3'...")
        bins = [-1, 9, 15, 20]
        labels = ['Fail', 'Pass', 'Excellent']
        data['G3_category'] = pd.cut(data['G3'], bins=bins, labels=labels)
    else:
        print("\nERROR FATAL: Kolom 'G3_category' dan 'G3' sama-sama tidak ada. File CSV Anda salah format.")
        exit()

# 6. Split features (X) dan target (y)
X = data.drop(columns=['G3_category', 'G3'], errors='ignore')
y = data['G3_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the Model
with mlflow.start_run():
    print("\nTraining the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained successfully! Accuracy: {accuracy:.2f}")
