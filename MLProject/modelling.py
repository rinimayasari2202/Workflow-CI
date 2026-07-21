import mlflow
import dagshub

# Inisialisasi DagsHub untuk GitHub Actions / CI
#dagshub.init(repo_owner='rinimayasari2202', repo_name='Workflow-CI', mlflow=True)

import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# 1. Setup MLflow Experiment
#mlflow.set_experiment("School_Performance_Optimization")

# 2. Load the preprocessed dataset
print("Loading data...")
try:
    data = pd.read_csv("student_cleaned_automated.csv")
except FileNotFoundError:
    print("ERROR: File CSV tidak ditemukan! Pastikan file sudah ada.")
    exit()

# FOOLPROOF FIX: Penanganan kolom target G3_category
if 'G3_category' not in data.columns:
    if 'G3' in data.columns:
        bins = [-1, 9, 15, 20]
        labels = ['Fail', 'Pass', 'Excellent']
        data['G3_category'] = pd.cut(data['G3'], bins=bins, labels=labels)
    else:
        print("ERROR FATAL: Kolom target tidak ditemukan.")
        exit()

# Split features (X) dan target (y)
X = data.drop(columns=['G3_category', 'G3'], errors='ignore')
y = data['G3_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_tune(X_train, y_train):
    # 3. Define the model and parameters for tuning
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # 4. Perform Hyperparameter Tuning
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

   # 5. Log to MLflow
    with mlflow.start_run() as run:
        best_model = grid_search.best_estimator_
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_accuracy", grid_search.best_score_)
        
        # CRITICAL: Save the model so it appears in the 'artifacts' folder
        mlflow.sklearn.log_model(best_model, "model")
        
        # SIMPAN RUN_ID KE FILE TEKS UNTUK DIBACA CI/CD
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        # -- FAIL-PROOF FIX: Simpan model secara lokal untuk Docker Build --
        import os
        import shutil
        if os.path.exists("local_model"):
            shutil.rmtree("local_model")
        mlflow.sklearn.save_model(best_model, "local_model")
        
        print(f"Model logged with Run ID: {run.info.run_id}")
        return run.info.run_id

# Eksekusi fungsi tuning
run_id = train_and_tune(X_train, y_train)