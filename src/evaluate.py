
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Hugging Face dataset repo
DATASET_REPO = "Murali0606/wellness_tourism_dataset"

def main():
    print("Loading test dataset from Hugging Face...")
    dataset = load_dataset(DATASET_REPO)
    test_df = dataset['test'].to_pandas()

    # Features and target
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']

    # Load trained model
    print("Loading trained Random Forest model...")
    model = joblib.load("models/random_forest.pkl")

    # Predict
    print("Running predictions...")
    preds = model.predict(X_test)

    # Evaluate
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }

    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics to file (for CI/CD logs)
    with open("models/evaluation_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}")

    print("Evaluation complete. Metrics saved to models/evaluation_metrics.txt")

if __name__ == "__main__":
    main()
