import logging
import pandas as pd
from titanic_pro.pipeline import pipeline, logger
from titanic_pro.utils import load_data
from joblib import load


def main():
    logger.info("Loading the model...")
    model = load("titanic_pro/models/model.joblib")

    logger.info("Loading test data...")
    _, test_df = load_data()

    logger.info("Predicting on test data...")
    predictions = model.predict(test_df)

    logger.info("Saving predictions...")
    submission = pd.DataFrame({"name": test_df["name"], "survived": predictions})
    submission.to_csv(
        "titanic_pro/submission.csv", index=False
    )  #  We would usually posted on a database instead


if __name__ == "__main__":
    main()
