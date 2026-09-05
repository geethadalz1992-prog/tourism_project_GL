
from pathlib import Path
import pandas as pd


DATA_PATH = Path("tourism_project/data/tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Columns available in CSV: {df.columns.tolist()}"
        )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset is empty.")

    validate_columns(df)

    print("DATASET REGISTERED SUCCESSFULLY")
    print("-" * 40)
    print(f"Dataset path     : {DATA_PATH}")
    print(f"Rows             : {df.shape[0]}")
    print(f"Columns          : {df.shape[1]}")
    print(f"Duplicate rows   : {df.duplicated().sum()}")
    print(f"Total null values: {df.isnull().sum().sum()}")

    print("\nTarget distribution: ProdTaken")
    print(df["ProdTaken"].value_counts().sort_index())


if __name__ == "__main__":
    main()
