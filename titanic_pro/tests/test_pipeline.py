import pytest
from titanic_pro.pipeline import pipeline
from titanic_pro.utils import load_data
from titanic_pro.custom_transformers import FamilySizeTransformer

def test_pipeline():
    train_df, _ = load_data()
    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']

    pipeline.fit(X, y)
    score = pipeline.score(X, y)

    assert score > 0.8, "Pipeline score is lower than expected"

def test_family_size_transformer():
    train_df, _ = load_data()
    X = train_df[['SibSp', 'Parch']]

    transformer = FamilySizeTransformer()
    family_size = transformer.fit_transform(X)

    assert not family_size.empty, "FamilySize is empty"
    assert 'FamilySize' in family_size.columns, "FamilySize column not found"