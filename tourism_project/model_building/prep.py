
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path("tourism_project/data/tourism.csv")
OUTPUT_DIR = Path("tourism_project/data/processed")

TARGET_COLUMN = "ProdTaken"
COLUMNS_TO_DROP = ["CustomerID"]


def main():
    # Load the dataset from the project data folder
    df = pd.read_csv(DATA_PATH)

    # Remove unnecessary identifier column
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Create 80% training data and 20% testing data
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET_COLUMN]
    )

    # Create output folder and save split CSV files locally
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Print summary
    print("Data preparation completed successfully.")
    print(f"Original data shape: {df.shape}")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Saved train file: {OUTPUT_DIR / 'train.csv'}")
    print(f"Saved test file: {OUTPUT_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
