import pytest
from titanic_pro.utils import load_data

# Not the ideal way to test the load_data function -- NEED TO MOCK!
def test_load_data():
    train_df, test_df = load_data()

    assert not train_df.empty, "Train data is empty"
    assert not test_df.empty, "Test data is empty"
