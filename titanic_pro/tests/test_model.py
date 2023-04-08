import pytest
from titanic_pro.train_utils import tune_model
from titanic_pro.utils import load_data
from titanic_pro.pipeline import pipeline
from sklearn.model_selection import train_test_split

def test_tune_model():
    train_df, _ = load_data()
    X = train_df.drop(columns=['survived'])
    y = train_df['survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params, best_score = tune_model(X_train, y_train)

    assert best_score > 0.8, "Model tuning score is lower than expected"