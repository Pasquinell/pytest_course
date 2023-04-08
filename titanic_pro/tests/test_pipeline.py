import pytest
from titanic_pro.pipeline import pipeline
from titanic_pro.utils import load_data

def test_pipeline():
    train_df, _ = load_data()
    X = train_df.drop(columns=['survived'])
    y = train_df['survived']

    pipeline.fit(X, y)
    score = pipeline.score(X, y)

    assert score > 0.8, "Pipeline score is lower than expected"