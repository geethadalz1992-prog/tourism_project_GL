from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Input files produced by prep.py
TRAIN_PATH = Path("tourism_project/data/processed/train.csv")
TEST_PATH = Path("tourism_project/data/processed/test.csv")

# Folder where the final deployable model will be stored
DEPLOYMENT_DIR = Path("tourism_project/deployment")
MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"

# Target column
TARGET_COLUMN = "ProdTaken"

# MLflow experiment tracking configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "tourism_package_prediction"


def load_data():
    """Load the prepared train and test CSV files."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Testing file not found: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def main():
    # 1. Load prepared train and test data
    train_df, test_df = load_data()

    # 2. Separate input features and target
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # 3. Identify numerical and categorical feature columns
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    # 4. Numeric preprocessing: fill missing values with median
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # 5. Categorical preprocessing: fill missing values and one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 6. Combine numeric and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 7. Create full model pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 8. Define Random Forest hyperparameter grid
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    # 9. Set local SQLite database as MLflow tracking backend
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Log data/run details
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("training_rows", len(train_df))
        mlflow.log_param("testing_rows", len(test_df))
        mlflow.log_param("number_of_features", X_train.shape[1])

        # 10. Tune the model using cross-validation F1 score
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        # 11. Get the best tuned pipeline
        best_model = grid_search.best_estimator_

        # 12. Test the model on unseen test data
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # 13. Calculate test performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        # 14. Log tuned parameters and evaluation metrics in MLflow
        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
                "test_roc_auc": roc_auc,
                "best_cv_f1_score": grid_search.best_score_,
            }
        )

        # 15. Save best model locally for deployment
        DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        # 16. Log model to MLflow.
        # Pickle avoids the skops trusted-types error.
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="tourism_purchase_model",
            serialization_format="pickle",
        )

        # 17. Display training/evaluation summary
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("-" * 50)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test precision: {precision:.4f}")
        print(f"Test recall: {recall:.4f}")
        print(f"Test F1 score: {f1:.4f}")
        print(f"Test ROC-AUC score: {roc_auc:.4f}")
        print(f"\nBest model saved at: {MODEL_PATH}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
