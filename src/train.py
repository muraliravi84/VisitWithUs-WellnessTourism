

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import json

# Hugging Face dataset repo
DATASET_REPO = "Murali0606/wellness_tourism_dataset"
MODEL_REPO = "Murali0606/wellness_tourism_model"

def main():
    print("Loading train/test datasets from Hugging Face...")
    dataset = load_dataset(DATASET_REPO)
    train_df = dataset['train'].to_pandas().drop(columns=["Unnamed: 0"], errors="ignore")
    test_df = dataset['test'].to_pandas().drop(columns=["Unnamed: 0"], errors="ignore")

    # Features and target
    X_train = train_df.drop(columns=['ProdTaken'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']

    # Identify categorical and numerical columns
    categorical_cols = [
        'TypeofContact', 'CityTier', 'Occupation', 'Gender',
        'PreferredPropertyStar', 'MaritalStatus', 'Designation',
        'ProductPitched'
    ]
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

    # Preprocessing: OneHot for categorical, passthrough for numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    # Random Forest parameters
    rf_params = {
        "n_estimators": 200,
        "random_state": 42,
        "class_weight": "balanced"
    }

    # Random Forest pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**rf_params))
    ])

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }
    print(f"Evaluation: {metrics}")

    # Log experiment parameters and metrics
    experiment_log = {
        "model": "RandomForestClassifier",
        "parameters": rf_params,
        "metrics": metrics
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/experiment_log.json", "w") as f:
        json.dump(experiment_log, f, indent=4)
    print("Experiment parameters and metrics logged to logs/experiment_log.json")

    # Save best model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")
    print("Random Forest model saved locally")

    # Push to Hugging Face Model Hub
    from huggingface_hub import HfApi, HfFolder, upload_file
    hf_token = HfFolder.get_token()
    api = HfApi()

    upload_file(
        path_or_fileobj="models/random_forest.pkl",
        path_in_repo="random_forest.pkl",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=hf_token
    )

    print(f"Model uploaded to Hugging Face Model Hub: {MODEL_REPO}")

if __name__ == "__main__":
    main()

