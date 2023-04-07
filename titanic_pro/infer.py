import logging
import pandas as pd
from titanic_pro.pipeline import pipeline, logger
from titanic_pro.utils import load_data
from joblib import load

def main():
    logger.info("Loading the model...")
    model = load('model.joblib')

    logger.info("Loading test data...")
    _, test_df = load_data()

    logger.info("Predicting on test data...")
    predictions = model.predict(test_df)

    logger.info("Saving predictions...")
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()