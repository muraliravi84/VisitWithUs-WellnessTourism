
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os

# Hugging Face dataset repo name
DATASET_REPO = "Murali0606/wellness_tourism_dataset"

def main():
    # 1. Load dataset from Hugging Face
    print(" Loading dataset from Hugging Face...")
    dataset = load_dataset(DATASET_REPO)
    df = dataset['train'].to_pandas()

    # 2. Data cleaning
    print(" Cleaning dataset...")
    # Drop unnecessary columns (example: CustomerID)
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Drop accidental index column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # 3. Train/test split
    print(" Splitting dataset...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # 4. Save locally (for debugging / CI logs)
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # 5. Push back to Hugging Face
    print(" Uploading processed dataset to Hugging Face...")
    dataset_dict.push_to_hub(DATASET_REPO)

    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()
