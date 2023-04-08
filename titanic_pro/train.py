import logging
import pandas as pd
from titanic_pro.pipeline import pipeline, logger
from titanic_pro.train_utils import tune_model
from titanic_pro.utils import load_data
from sklearn.model_selection import train_test_split
from joblib import dump


def main():
    logger.info("Loading data...")
    train_df, _ = load_data()
    logger.info(train_df.columns.values.tolist())
    y = train_df["survived"]
    X = train_df.drop(columns=["survived"])

    logger.info("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Tuning model...")
    best_params, best_score = tune_model(X_train, y_train)
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score}")

    logger.info("Training the model with the best parameters...")
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating the model on the validation set...")
    score = pipeline.score(X_val, y_val)
    logger.info(f"Validation accuracy: {score}")

    logger.info("Saving the model...")
    dump(pipeline, "titanic_pro/models/model.joblib")


if __name__ == "__main__":
    main()
